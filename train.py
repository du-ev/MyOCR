import numpy as np
import torch
import torch.nn as nn
from torch import autocast, GradScaler
import time
from torch.utils.data import DataLoader
import torch.optim as optim

import os
import glob

import torch.nn.functional as F
import re

from crnn import CRNN
from dataset import CRNNDataset
from tqdm.auto import tqdm

import torchvision.transforms.functional as TF
import random
import optuna

from hyperparameters import hparams

alphabet = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{}~£°—‘’“”"
blank_idx   = 0
char_to_int = {ch: i + 1 for i, ch in enumerate(alphabet)}  # 1..N
idx_to_char = {i: ch for ch, i in char_to_int.items()} # inverse
num_classes = len(alphabet) + 1  
# not a "space", but a token that lets the model separate letters. output from B-LSTM might be like hheelllloo -> helo (we lose an L). 
# versus with CTC blank token \: ---hh---ee--ll---ll--o -> hello

# batch_size = 32
# epochs = 50
# learning_rate = 0.1e-4
# weight_dec = 1e-4

force_load = False

train_txt_path = "combined_train_data.txt"
val_txt_path = "combined_val_data.txt"

checkpoint_dir = "checkpoints"

def train_and_evaluate(config, trial=None):
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = CRNNDataset(pt_dir="pt/train", alphabet=alphabet)
    val_data = CRNNDataset(pt_dir="pt/val", alphabet=alphabet)

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch,
        shuffle=True,
        num_workers=8,
        pin_memory=True, #supposed to speed up moving data tensors to gpu .to(device)
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn 
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_fn
    )

    crnn = CRNN(num_classes, config.hidden, dropout=config.dropout).to(device, non_blocking=True)
    criterion = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.AdamW(crnn.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = GradScaler(device="cuda")

    if not trial:
        start_epoch = load_ocr_model(crnn, checkpoint_dir, force_load)
    else:
        start_epoch = 0
    config.epochs += start_epoch

    best_loss = float("inf")

    for epoch in range(start_epoch, config.epochs):
        crnn.train() #sets model train mode (dropout, etc active)
        loss_total = 0.0

        correct_train = 0
        total_train = 0

        # images: 4d tensor image data [batch_size, channels, height, max_width (padded)] 
        # input_lengths: 1d encoded labels for the batch
        # input_lengths: 1d tensor, sequence length for each item in batch 
        # label_lengths: 1d tensor, length of label for each image in batch
        for images, labels, label_lengths, widths in tqdm(train_loader, desc=f"epoch {epoch+1}/{config.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            label_lengths = label_lengths.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = None

            with autocast(device_type="cuda"):
                logits = crnn(images)
                logits = F.log_softmax(logits, dim=2)
                
                T, B, _ = logits.shape      # T=time-steps, B=batch-size
                input_lengths = torch.full(
                    (B,),         # one length per batch element
                    T,            # use every time-step
                    dtype=torch.long,
                    device=logits.device
                )
                loss = criterion(logits, labels, input_lengths, label_lengths)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_total += loss.item()

            with torch.no_grad():
                preds = ctc_decode(logits.detach(), blank_idx, input_lengths)
                true_captions = tensor_to_strings(labels.cpu(), label_lengths.cpu())
                correct_train += sum(p == t for p, t in zip(preds, true_captions))
                total_train += len(true_captions)


        avg_loss = loss_total/len(train_loader)
        avg_acc = correct_train / total_train
        if not trial:
            print(f"epoch {epoch+1}/{config.epochs} train loss: {avg_loss:.4f} | train acc: {avg_acc:.4f}")
        
        #Validation Loop
        crnn.eval()
        val_loss_total = 0.0

        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels, label_lengths, widths in tqdm(val_loader, desc=f"validating {epoch+1}/{config.epochs}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                logits = crnn(images)
                logits = F.log_softmax(logits, dim=2)

                input_lengths = torch.clamp((widths // 16) * 4, max=logits.size(0)).to(device, non_blocking=True)

                loss = criterion(logits, labels, input_lengths, label_lengths)
                val_loss_total += loss.item()

                with torch.no_grad():
                    preds = ctc_decode(logits.detach(), blank_idx, input_lengths)
                    true_captions = tensor_to_strings(labels.cpu(), label_lengths.cpu())
                    correct_val += sum(p == t for p, t in zip(preds, true_captions))
                    total_val += len(true_captions)

        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_acc = correct_val / total_val

        if trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        else:
            save_model(crnn, f"checkpoints/best_model_epoch{epoch + 1}.pt")
            print(f"epoch {epoch+1}/{config.epochs} validation test: {avg_val_loss:.4f} | val acc: {avg_val_acc:.4f}")
        best_loss = min(best_loss, avg_val_loss)
    
    return best_loss

def objective(trial):
    wd = trial.suggest_categorical(
        "weight_decay",
        [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    )
    config = hparams(
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        weight_decay = wd,
        hidden = trial.suggest_int("hidden", 128, 512, step=64),
        dropout = trial.suggest_float("dropout", 0.1, 0.6),
        optimiser = trial.suggest_categorical("optimiser", ["adam", "adamw"]),
        scheduler = trial.suggest_categorical("scheduler", ["cosine", "step"]),
        batch = trial.suggest_categorical("batch", [32, 64, 96]),
        epochs = 15,
        seed = trial.suggest_int("seed", 1, 10000)
    )
    torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)

    train_and_evaluate(config, trial)


        
def ctc_decode(logits, blank_idx, input_lengths=None): #returns list of decoded strings
    #logits: [B, T, num_chars]
    #blank_idx: index of ctc blank
    logits = logits.permute(1, 0, 2)
    preds = logits.argmax(dim=2)

    results=[]
    for i, pred in enumerate(preds):
        real_width = input_lengths[i] if input_lengths is not None else pred.size(0)
        word = []
        prev = blank_idx
        for idx in pred:
            idx = idx.item()
            if idx != blank_idx and idx != prev:
                word.append(idx_to_char[idx])
            prev = idx
        results.append("".join(word))
    return results

def ctc_decode_beam(logits, beam_width=5):
    logp = logits.log_softmax(2).squeeze(1)
    beam = [(("", blank_idx), 0.0)]
    
    for t in range(logp.size(0)):
        candidates = []
        for (pref, last), score in beam:
            for c, lp in enumerate(logp[t]):
                sc = score + lp.item()
                new_pref = pref if (c == blank_idx or c == last) else pref + idx_to_char[c]
                candidates.append(((new_pref, c), sc))
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return beam[0][0]

def load_ocr_model(model, ckpt_dir, force=True):
    start = 0
    if force:
        pts = glob.glob(os.path.join(ckpt_dir, "*.pt"))
        if pts:
            latest = max(pts, key=os.path.getmtime)
            model.load_state_dict(torch.load(latest, map_location="cpu"))
            st = re.search(r"epoch(\d+)", latest)
            if st:
                start = int(st.group(1))
            print(f"loaded model weights from {latest}")
        else:
            print(f"no .pt files found in {ckpt_dir}")
    else:
        for f in glob.glob(os.path.join(ckpt_dir, "*.pt")):
            os.remove(f)
    return start

def tensor_to_strings(flat_labels, lengths):
    out, idx = [], 0
    for length in lengths:
        seq = flat_labels[idx : idx + length].tolist()
        out.append("".join(idx_to_char[i] for i in seq))
        idx += length
    return out

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    weights = torch.load(path, None)
    model.load_state_dict(weights)
    return model

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch: # if this batch had 8 empty samples or something
        return None, None, None
    # images of variable width cannot be stacked into the same batch... we gotta pad them. apparently ctc loss is okay with this if they are all left aligned (padded to right)
    # inspiration from https://www.codefull.org/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/ 
    images, labels = zip(*batch)

    def jitter(t):
        if random.random() < 0.3:
            angle = random.gauss(0, 5)
            t = TF.rotate(t, angle, expand=False)
        if random.random() < 0.3:
            factor = 0.8 + 0.4 * random.random()
            t = TF.adjust_contrast(t, factor)
        return t
    images = [jitter(img) for img in images]

    max_width = max(img.shape[2] for img in images)
    padded_images = []

    for img in images:
        padding = (0, max_width - img.shape[2], 0, 0)
        padded_img = nn.functional.pad(img, padding, "constant", 0)
        padded_images.append(padded_img)
    images_tensor = torch.stack(padded_images, 0)

    labels_tensor = torch.cat(labels)
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    # output_length = max_width // 29 # found from testing_debug.py. approximation
    # input_lengths = torch.full(size=(len(images),), fill_value=output_length, dtype=torch.long)

    widths_tensor = torch.tensor([img.shape[2] for img in images], dtype=torch.long)

    return images_tensor, labels_tensor, label_lengths, widths_tensor

if __name__ == '__main__':
    # config = hparams()
    # train_and_evaluate(config)
    study = optuna.create_study(direction="minimize", storage="sqlite:///crnn_optuna.db", load_if_exists=True, pruner=optuna.pruners.MedianPruner(n_startup_trials=10,                                                               n_warmup_steps=3))
    study.optimize(objective, n_trials=100, timeout=4*60*60)