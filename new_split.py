import os
import shutil
import random
from pathlib import Path

# ==========================
# Paths d'entrée et de sortie
# ==========================

root_imgs = Path("/home/usrs/hnoel/RAVIR_dataset/train/training_images")
root_masks = Path("/home/usrs/hnoel/RAVIR_dataset/train/training_masks")

out_root = Path("/home/usrs/hnoel/SPLIT_RAVIR")

# Structure des dossiers de sortie
splits = {
    "Test": (5),
    "Train": (14),
    "Val": (4),
}

# ==========================
# Préparation des dossiers
# ==========================

for split in splits:
    (out_root / split / "Images").mkdir(parents=True, exist_ok=True)
    (out_root / split / "Masks").mkdir(parents=True, exist_ok=True)

# ==========================
# Récupération des fichiers
# ==========================

img_files = sorted(list(root_imgs.glob("IR_Case_*.png")))

# Sanity check
assert len(img_files) == 23, f"Expected 23 images, found {len(img_files)}"

# Mélanger pour random split
random.seed(42)
random.shuffle(img_files)

# ==========================
# Split
# ==========================

test_imgs  = img_files[:5]
train_imgs = img_files[5:5+14]
val_imgs   = img_files[5+14:]

assert len(val_imgs) == 4, "Val split incorrect"

splits_dict = {
    "Test": test_imgs,
    "Train": train_imgs,
    "Val": val_imgs,
}

# ==========================
# Copie des fichiers
# ==========================

for split_name, img_list in splits_dict.items():
    print(f"Processing {split_name} ({len(img_list)} images)...")
    
    for img_path in img_list:
        fname = img_path.name
        mask_path = root_masks / fname

        # Vérification que le masque existe
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {fname}")

        # Destination
        dest_img = out_root / split_name / "Images" / fname
        dest_mask = out_root / split_name / "Masks" / fname

        shutil.copy(img_path, dest_img)
        shutil.copy(mask_path, dest_mask)

print("\nDone! Split created in:", out_root)
