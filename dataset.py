import torch
from torch.utils.data import Dataset
import cv2
import ast
import os

from image_normalization import load_image
from preprocess_datasets import convert_image

class CRNNDataset(Dataset): # extend dataset
    def __init__(self, pt_dir, alphabet):
        self.files = sorted(os.path.join(pt_dir, fn) for fn in os.listdir(pt_dir) if fn.endswith(".pt"))

        self.alphabet = alphabet

        # "encoder" and our "decoder" for CTC purposes e.g. "cat" -> [3, 1, 20]
        self.char_to_int = {char: i + 1 for i, char in enumerate(self.alphabet)} # i + 1 because CTC reserves index 0 for blank token to properly convert rnn output to text

    def __getitem__(self, index):
        temp = torch.load(self.files[index], map_location="cpu")
        image = temp["image"].float()
        label_str = temp["label"]

        encoded = [self.char_to_int[c] for c in label_str if c in self.char_to_int]
        return image, torch.tensor(encoded, dtype=torch.long)
    
    def __len__(self):
        return len(self.files)