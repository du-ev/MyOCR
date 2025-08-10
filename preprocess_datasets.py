import numpy as np
import os
from image_normalization import grayscale_image, sharpen_image, threshold_image, align_text,  load_image, remove_noise, to_white_on_black
import cv2
from typing import List, Tuple
import json
import kagglehub

from tqdm.auto import tqdm

data_height = 64
# if changed, take a look at stride in crnn.py also

def convert_image(img_array):
    img = grayscale_image(img_array)
    # img = sharpen_image(img)
    # img = remove_noise(img)
    # img = threshold_image(img) # the r, u, m, n all blend together
    # img = align_text(img)
    
    img = to_white_on_black(img)

    if img.ndim != 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return np.array([]) # Return an empty array for invalid images

    h, w = img.shape[:2]    
    new_width = int(w * (data_height / h))

    if new_width <= 0:
        new_width = 1

    img = cv2.resize(img, (new_width, data_height), interpolation=cv2.INTER_AREA)

    img_3_channel = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img_3_channel / 255.0 # normalize pixel range to [0, 1]

def test_convert_image(img_array):
    img = grayscale_image(img_array)
    cv2.imshow("grayscale_image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    img = sharpen_image(img)
    cv2.imshow("sharpen_image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    img = threshold_image(img)
    cv2.imshow("threshold_image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # img = align_text(img)
    img = remove_noise(img)
    img = to_white_on_black(img)
    cv2.imshow("to_white_on_black", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    h, w = img.shape[:2]
    new_width = int(w * (data_height / h))
    img = cv2.resize(img, (new_width, data_height), interpolation=cv2.INTER_AREA)

    return img / 255.0

def preprocess_mnist(dataset_path) -> List[Tuple[str, str]]:
    """ processes MNIST dataset, returns list of (path, label) """
    preprocessed_data = []
    
    label_file = os.path.join(dataset_path, 'v011_labels_small.json')
    images_dir = os.path.join(dataset_path, 'dataset', 'v011_words_small')

    with open(label_file, "r", encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    for filename, label in tqdm(labels_dict.items(), desc="processing MNIST dataset"):
        img_path = os.path.join(images_dir, filename)

        if os.path.exists(img_path):
            preprocessed_data.append((img_path, label))
            
    return preprocessed_data

def preprocess_gnhk(dataset_path, manifest_path) -> List[Tuple[str, str, Tuple]]:
    """ processes GNHK dataset, returns list of (path, label, bbox) """
    preprocessed_data = []
    image_files = {f.name: f.path for f in os.scandir(dataset_path) if f.is_file()}

    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="processing GNHK dataset"):
        entry = json.loads(line)
        img_basename = os.path.basename(entry['source-ref'])

        if img_basename in image_files:
            full_image_path = image_files[img_basename]

            for t in entry['annotations']['texts']:
                label = t['text']
                polygon = t['polygon']

                xs = [p["x"] for p in polygon]
                ys = [p["y"] for p in polygon]

                bbox = (min(xs), min(ys), max(xs), max(ys))
                preprocessed_data.append((full_image_path, label, bbox))

    return preprocessed_data

def create_combined_data(force_regenerate_data = False):
    train_txt_path = "combined_train_data.txt"
    val_txt_path = "combined_val_data.txt"

    if not force_regenerate_data and os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
        print(f"{train_txt_path} and {val_txt_path} already exists, skipping")
        return
    
    # will download datasets if needed to ~\.cache\kagglehub\datasets
    mnist_path = kagglehub.dataset_download('backalla/words-mnist')
    gnhk_train_path = os.path.join(kagglehub.dataset_download('evandu/GNHK-dataset'), "gnhk/train_data/train")
    gnhk_val_path = os.path.join(kagglehub.dataset_download('evandu/GNHK-dataset'), "gnhk/test_data/test")

    # initialize and split MNIST
    def split(ds, frac=0.8):
        np.random.shuffle(ds)
        cut = int(frac * len(ds))
        return ds[:cut], ds[cut:]

    mnist_train_data, mnist_val_data = split(preprocess_mnist(mnist_path))

    # initialize GNHK train data
    gnhk_train_manifest = os.path.join(gnhk_train_path, "train.manifest")
    gnhk_full = preprocess_gnhk(gnhk_train_path, gnhk_train_manifest) + preprocess_gnhk(gnhk_val_path,   os.path.join(gnhk_val_path, "test.manifest"))
    gnhk_train_data, gnhk_val_data = split(gnhk_full)
    
    # initialize GNHK validation data
    gnhk_val_manifest = os.path.join(gnhk_val_path, "test.manifest")
    gnhk_val_data = preprocess_gnhk(gnhk_val_path, gnhk_val_manifest)

    # turn into combined train/val da ta
    combined_train_data = mnist_train_data + gnhk_train_data
    combined_val_data = mnist_val_data + gnhk_val_data

    np.random.shuffle(combined_train_data)
    np.random.shuffle(combined_val_data)

    # writes them to txt: "path|caption|(None for mnist OR bbox 4 coords for gnhk)"
    with open(train_txt_path, "w", encoding="utf-8") as f:
        for item in tqdm(combined_train_data, desc="Writing train labels"):
            f.write(f"{item[0]}|{item[1]}|{item[2] if len(item) == 3 else 'None'}\n")
            
    with open(val_txt_path, "w", encoding="utf-8") as f:
        for item in tqdm(combined_val_data, desc="Writing val labels"):
            f.write(f"{item[0]}|{item[1]}|{item[2] if len(item) == 3 else 'None'}\n")

if __name__ == "__main__":
    img = cv2.imread("image.png")
    cv2.imshow("original image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    img = test_convert_image(img)
    cv2.imshow("final image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()