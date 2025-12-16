from pathlib import Path
import cv2

ROOT = Path("/home/usrs/hnoel/CHASE_DB1")
TARGET = 960


def crop_center_to_960(img):
    h, w = img.shape[:2]

    if h < TARGET or w < TARGET:
        raise ValueError(f"Image trop petite : {h}x{w}")

    y0 = (h - TARGET) // 2
    x0 = (w - TARGET) // 2

    return img[y0:y0 + TARGET, x0:x0 + TARGET]


def main():
    for png_path in ROOT.rglob("*.png"):
        img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Impossible de lire {png_path}")
            continue

        h, w = img.shape[:2]

        if (h, w) == (TARGET, TARGET):
            continue  # déjà OK

        try:
            cropped = crop_center_to_960(img)
        except Exception as e:
            print(f"[ERROR] {png_path} : {e}")
            continue

        cv2.imwrite(str(png_path), cropped)
        print(f"[OK] {png_path} : {h}x{w} -> 960x960")


if __name__ == "__main__":
    main()
