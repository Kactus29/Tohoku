"""
Minimal script converting first_tests.ipynb into a runnable .py:
- crée une séparation train/val (try_1) si nécessaire
- définit le dataset RAVIRDataset
- construit UNet
- entraîne la version "weighted + hard penalty"
- évalue et sauvegarde prédictions (5 images de validation)
Notes:
- Paths are relative to this file's parent directory.
- Adjust EPOCHS_W, BATCH_SIZE, LR etc. as needed.
"""
from pathlib import Path
from shutil import copy2
import random
import math
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# ----------------------
# Config / paths
# ----------------------
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
RAVIR_TRAIN = PARENT / 'RAVIR Dataset' / 'train'  # expected location of original dataset
TRY_DIR = PARENT / 'try_1'                         # destination for split
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
LR = 1e-4
NUM_CLASSES = 3
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------
# Helper: create random split (copies images + masks)
# ----------------------
def create_random_split(ravir_train_dir: Path, dest: Path, seed=42):
    imgs_dir = ravir_train_dir / 'training_images'
    masks_dir = ravir_train_dir / 'training_masks'
    if dest.exists():
        print(f"'{dest}' exists — split skipped.")
        return
    imgs = sorted(imgs_dir.glob('*.png'))
    cases = [p.stem for p in imgs]
    random.seed(seed)
    random.shuffle(cases)
    if len(cases) >= 23:
        n_train = 18
    else:
        n_train = int(round(0.8 * len(cases)))
    train_cases = cases[:n_train]
    val_cases = cases[n_train:]
    train_dest = dest / 'train'
    val_dest = dest / 'val'
    for d in [train_dest / 'training_images', train_dest / 'training_masks',
              val_dest / 'training_images', val_dest / 'training_masks']:
        d.mkdir(parents=True, exist_ok=True)

    def copy_pairs(case_list, dest_base):
        for stem in case_list:
            img_src = imgs_dir / (stem + '.png')
            mask_src = masks_dir / (stem + '.png')
            if img_src.exists():
                copy2(img_src, dest_base / 'training_images' / img_src.name)
            else:
                print(f'Missing image: {img_src}')
            if mask_src.exists():
                copy2(mask_src, dest_base / 'training_masks' / mask_src.name)
            else:
                print(f'Missing mask: {mask_src}')

    copy_pairs(train_cases, train_dest)
    copy_pairs(val_cases, val_dest)
    print(f'Copied {len(train_cases)} train and {len(val_cases)} val cases into {dest}')

# ----------------------
# Dataset
# ----------------------
COLOR_TO_LABEL = {
    (0,0,0): 0,
    (255,255,255): 1,
    (128,128,128): 2
}

class RAVIRDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=IMG_SIZE):
        self.img_paths = sorted(list(Path(img_dir).glob('*.png')))
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_p = self.img_paths[idx]
        mask_p = self.mask_dir / img_p.name
        img = Image.open(img_p).convert('RGB').resize(self.img_size, Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2,0,1))  # C,H,W
        img_t = torch.from_numpy(img).float()
        # load mask and map colors to labels
        m = Image.open(mask_p).convert('RGB').resize(self.img_size, Image.NEAREST)
        m_arr = np.array(m, dtype=np.uint8)
        label = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
        for color, lab in COLOR_TO_LABEL.items():
            mask = np.all(m_arr == np.array(color, dtype=np.uint8), axis=-1)
            label[mask] = lab
        label_t = torch.from_numpy(label).long()  # H,W
        return img_t, label_t

# ----------------------
# UNet model (kept similar to notebook)
# ----------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, in_channels=3, features=[32,64,128,256]):
        super().__init__()
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for f in features:
            self.encs.append(DoubleConv(in_channels, f))
            self.pools.append(nn.MaxPool2d(2))
            in_channels = f
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.upconvs = nn.ModuleList()
        self.decs = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.decs.append(DoubleConv(f*2, f))
        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decs, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        x = self.final_conv(x)
        return x

# ----------------------
# Utilities: metrics, colorize
# ----------------------
def compute_mean_iou(preds, labels, num_classes=NUM_CLASSES):
    preds = preds.view(-1)
    labels = labels.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return np.nanmean(np.array(ious, dtype=np.float32))

COLOR_MAP = {0: (0,0,0), 1: (255,255,255), 2: (128,128,128)}
def colorize_mask(mask):
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for k,v in COLOR_MAP.items():
        out[mask==k] = v
    return out

# ----------------------
# Training loop: weighted + hard penalty as in notebook
# ----------------------
def train_weighted_hard(train_loader, val_loader, save_dir, epochs=30, lr=LR, lambda_penalty=4.0):
    model_w = UNet().to(DEVICE)
    class_weights = torch.tensor([1.0, 5.0, 5.0], device=DEVICE)
    criterion_w = nn.CrossEntropyLoss(weight=class_weights)
    optimizer_w = optim.Adam(model_w.parameters(), lr=lr)
    history = {'train_loss':[], 'val_loss':[], 'train_miou':[], 'val_miou':[]}
    for epoch in range(1, epochs+1):
        model_w.train()
        running_loss = 0.0
        miou_sum = 0.0
        batches = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer_w.zero_grad()
            outputs = model_w(imgs)  # logits N,C,H,W
            ce = criterion_w(outputs, masks)
            probs = F.softmax(outputs, dim=1)
            pos_mask = (masks > 0).float()   # protect class 1/2
            prob0 = probs[:, 0, :, :]
            pos_count = pos_mask.sum()
            if pos_count.item() > 0:
                penalty = lambda_penalty * (prob0 * pos_mask).sum() / pos_count
            else:
                penalty = torch.tensor(0.0, device=DEVICE)
            loss = ce + penalty
            loss.backward()
            optimizer_w.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            miou_sum += compute_mean_iou(preds.cpu(), masks.cpu())
            batches += 1
        train_loss = running_loss / len(train_loader.dataset) if len(train_loader.dataset)>0 else 0
        train_miou = miou_sum / batches if batches>0 else 0

        # validation
        model_w.eval()
        val_loss = 0.0
        miou_sum = 0.0
        batches = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model_w(imgs)
                ce = criterion_w(outputs, masks)
                probs = F.softmax(outputs, dim=1)
                pos_mask = (masks > 0).float()
                prob0 = probs[:, 0, :, :]
                pos_count = pos_mask.sum()
                penalty = (lambda_penalty * (prob0 * pos_mask).sum() / pos_count) if pos_count.item()>0 else torch.tensor(0.0, device=DEVICE)
                loss = ce + penalty
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                miou_sum += compute_mean_iou(preds.cpu(), masks.cpu())
                batches += 1
        val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset)>0 else 0
        val_miou = miou_sum / batches if batches>0 else 0
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_miou'].append(train_miou)
        history['val_miou'].append(val_miou)
        print(f'[weighted-hard] Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} train_miou: {train_miou:.4f} val_miou: {val_miou:.4f}')

    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f'unet_pytorch_weighted_hard_{epochs}ep.pth'
    torch.save(model_w.state_dict(), str(model_path))
    print('Saved weighted-hard model to', model_path)
    return model_w, history

# ----------------------
# Evaluation: save colored predictions for first N val images and print metrics
# ----------------------
def evaluate_and_save_preds(model, ds, out_dir, max_images=5):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    total_pixels = 0
    correct_pixels = 0
    class_intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_unions = np.zeros(NUM_CLASSES, dtype=np.float64)
    n = min(max_images, len(ds))
    for idx in range(len(ds)):
        img_t, mask_t = ds[idx]
        img = img_t.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(img)
            pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        gt = mask_t.numpy().astype(np.uint8)
        total_pixels += gt.size
        correct_pixels += (pred == gt).sum()
        for c in range(NUM_CLASSES):
            pred_c = (pred == c)
            gt_c = (gt == c)
            inter = np.logical_and(pred_c, gt_c).sum()
            union = np.logical_or(pred_c, gt_c).sum()
            class_intersections[c] += inter
            class_unions[c] += union
        if idx < n:
            pred_col = colorize_mask(pred)
            Image.fromarray(pred_col).save(out_dir / f'pred_{idx}.png')
    pixel_acc = correct_pixels / total_pixels if total_pixels>0 else 0
    ious = []
    for c in range(NUM_CLASSES):
        if class_unions[c] == 0:
            ious.append(float('nan'))
        else:
            ious.append(class_intersections[c] / class_unions[c])
    mean_iou = np.nanmean(ious)
    print('Validation results:')
    print(f' Pixel Accuracy: {pixel_acc:.4f}')
    for c, iou in enumerate(ious):
        print(f'  Class {c} IoU: {iou if not np.isnan(iou) else "n/a"}')
    print(f' Mean IoU: {mean_iou:.4f}')
    return {'pixel_acc': pixel_acc, 'ious': ious, 'mean_iou': mean_iou}

# ----------------------
# Main
# ----------------------
def main():
    # 1) make split if needed
    if not (PARENT / 'try_1').exists():
        if not RAVIR_TRAIN.exists():
            print('RAVIR dataset not found at', RAVIR_TRAIN)
            return
        create_random_split(RAVIR_TRAIN, TRY_DIR, seed=SEED)

    # 2) prepare datasets / loaders
    train_imgs_dir = TRY_DIR / 'train' / 'training_images'
    train_masks_dir = TRY_DIR / 'train' / 'training_masks'
    val_imgs_dir = TRY_DIR / 'val' / 'training_images'
    val_masks_dir = TRY_DIR / 'val' / 'training_masks'
    print('Train images folder exists:', train_imgs_dir.exists())
    print('Val images folder exists:', val_imgs_dir.exists())

    train_ds = RAVIRDataset(train_imgs_dir, train_masks_dir, IMG_SIZE)
    val_ds = RAVIRDataset(val_imgs_dir, val_masks_dir, IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3) train weighted-hard model
    save_dir = TRY_DIR
    # smaller epochs by default; increase if you want
    EPOCHS_W = 30
    model_w, history_w = train_weighted_hard(train_loader, val_loader, save_dir, epochs=EPOCHS_W, lr=LR, lambda_penalty=4.0)

    # 4) evaluate on validation (save first 5 preds)
    preds_out = TRY_DIR / 'predictions_on_val'
    metrics = evaluate_and_save_preds(model_w, val_ds, preds_out, max_images=5)
    print('Predictions saved to', preds_out)

if __name__ == '__main__':
    main()