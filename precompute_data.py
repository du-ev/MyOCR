import numpy as np
import os, ast, torch, concurrent.futures
from tqdm.auto import tqdm
from preprocess_datasets import load_image, convert_image

def _process(args):
    index, line, out_dir = args
    try:
        img_path, label, bbox = line.split("|")
        img = load_image(img_path)

        if bbox and bbox != 'None':
            try:
                bbox = ast.literal_eval(bbox)
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    left, upper, right, lower = map(int, bbox)
                    img = img[upper:lower, left:right]
            except Exception:
                pass

        img = convert_image(img)
        if img is None or img.size == 0:
            return  # skip

        if img.ndim == 2:
            img = img[:, :, None]
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        file_path = os.path.join(out_dir, f"{index:08d}.pt")
        torch.save({"image": tensor, "label": label}, file_path)
    except Exception:
        print("\nskipped bad sample")


def precompute(caption_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(caption_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        list(
            tqdm(
                pool.map(_process, ((i, ln, out_dir) for i, ln in enumerate(lines))),
                total=len(lines),
                desc=f"precomputing {caption_path}",
            )
        )


if __name__ == "__main__":
    precompute("combined_train_data.txt", "pt/train")
    precompute("combined_val_data.txt",   "pt/val")
