from tqdm.auto import tqdm

lines = open("combined_train_data.txt", encoding="utf-8").read().splitlines()

chars = set()

for line in tqdm(lines, desc="generating alphabet"):
    #C:\Users\EvanD\.cache\kagglehub\datasets\evandu\GNHK-dataset\versions\1\gnhk/train_data/train\eng_EU_175.jpg|hope|(406, 1348, 736, 1613)
    parts = line.split("|")
    caption = parts[1]
    for c in caption:
        chars.add(c)
alphabet = "".join(sorted(chars))
print(alphabet)