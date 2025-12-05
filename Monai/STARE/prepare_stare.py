import os
import random
from PIL import Image

# --- CONFIG ---
ROOT = "/home/usrs/hnoel/STARE"                    # dossier original
OUT = "/home/usrs/hnoel/STARE_704"                 # nouveau dataset
ANNOTATOR = "ah"                  # "ah" ou "vk"
NEW_SIZE = (704, 704)
TRAIN_RATIO = 0.5                 # 50-50 train-test split
# ----------------

os.makedirs(f"{OUT}/training/images", exist_ok=True)
os.makedirs(f"{OUT}/training/masks", exist_ok=True)
os.makedirs(f"{OUT}/test/images", exist_ok=True)
os.makedirs(f"{OUT}/test/masks", exist_ok=True)

# récupérer liste des images originales
image_files = sorted(os.listdir(f"{ROOT}/stare-images"))
image_files = [f for f in image_files if f.endswith(".ppm")]

# shuffle pour split aléatoire
random.shuffle(image_files)
split_idx = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:split_idx]
test_files = image_files[split_idx:]

def load_ppm(path):
    return Image.open(path).convert("RGB")

def load_mask(path):
    # NOTE : STARE masks are PPM but binary colors → convert to single channel
    return Image.open(path).convert("L")

def save_resized(im, path):
    im = im.resize(NEW_SIZE, Image.BILINEAR)
    im.save(path)

# --- PROCESS ---
for dataset, files in [("training", train_files), ("test", test_files)]:
    for fname in files:
        base = fname.replace(".ppm", "")

        img_path = f"{ROOT}/stare-images/{fname}"
        mask_path = f"{ROOT}/labels-{ANNOTATOR}/{base}.{ANNOTATOR}.ppm"

        # load
        img = load_ppm(img_path)
        mask = load_mask(mask_path)

        # resize and save to PNG
        save_resized(img, f"{OUT}/{dataset}/images/{base}.png")
        save_resized(mask, f"{OUT}/{dataset}/masks/{base}_mask.png")

print("STARE 704×704 dataset created successfully!")
