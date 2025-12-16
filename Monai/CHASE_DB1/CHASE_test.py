from pathlib import Path
import cv2
from collections import Counter

ROOT = Path("/home/usrs/hnoel/CHASE_DB1")


def main():
    sizes = Counter()

    for png_path in ROOT.rglob("*.png"):
        img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Impossible de lire {png_path}")
            continue

        h, w = img.shape[:2]
        sizes[(h, w)] += 1
        print(f"{png_path} -> {h}x{w}")

    print("\n=== RÉSUMÉ DES TAILLES ===")
    for size, count in sizes.items():
        print(f"{size[0]}x{size[1]} : {count} fichiers")


if __name__ == "__main__":
    main()
