import os
from pathlib import Path
from PIL import Image

SRC_DIR = Path("/home/usrs/hnoel/DRIVE")
DST_DIR = Path("/home/usrs/hnoel/DRIVE_608")
TARGET_SIZE = (608, 608)  # (W, H)


# ==========================
# Directory creation
# ==========================
def ensure_dirs():
    for split in ["training", "test"]:
        for sub in ["images", "mask", "1st_manual", "fov"]:
            (DST_DIR / split / sub).mkdir(parents=True, exist_ok=True)


ensure_dirs()


# ==========================
# Resize helper
# ==========================
def resize_image(path_in, path_out, mode):
    img = Image.open(path_in)

    if mode == "rgb":
        img = img.resize(TARGET_SIZE, resample=Image.BILINEAR)
    else:
        img = img.resize(TARGET_SIZE, resample=Image.NEAREST)

    img.save(path_out)


# ==========================
# Process split
# ==========================
def process_split(split):
    print(f"=== Processing {split.upper()} ===")

    base = SRC_DIR / split

    # detect whether directory is "mask" or "masks"
    mask_dir = None
    if (base / "mask").exists():
        mask_dir = "mask"
    elif (base / "masks").exists():
        mask_dir = "masks"

    for folder, mode in [
        ("images", "rgb"),
        (mask_dir, "mask"),              # now robust
        ("1st_manual", "mask"),
        ("fov", "mask") if (base / "fov").exists() else (None, None),
    ]:
        if folder is None:
            continue

        src = base / folder
        dst = DST_DIR / split / folder
        dst.mkdir(parents=True, exist_ok=True)

        for file in sorted(src.iterdir()):
            if not file.is_file():
                continue

            path_out = dst / file.name
            resize_image(file, path_out, mode)
            print(f"✔ {file} → {path_out}")


# ==========================
# RUN
# ==========================
process_split("training")
process_split("test")

print("\n🎉 DONE: All DRIVE images resized to 608×608 and fully copied.")
