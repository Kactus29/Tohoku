import os
import numpy as np
from PIL import Image

x = str(input("Enter the name of the experience folder"))

labels_dir = f"/home/usrs/hnoel/IIC/nnUNetv2/{x}"

for filename in os.listdir(labels_dir):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(labels_dir, filename)

    img = np.array(Image.open(path))

    # Create new label image
    new = np.zeros_like(img, dtype=np.uint8)

    new[img == 0] = 0        # background
    new[img == 1] = 255      # artères
    new[img == 2] = 128      # veines

    Image.fromarray(new).save(path)

    print("Fixed:", filename)

print("✔ Conversion terminée.")


# Lol