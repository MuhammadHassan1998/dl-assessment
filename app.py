"""
Starter notebook: Partial-supervision Segmentation (PyTorch)

Files included in this single script/notebook:
- Setup & requirements
- Utilities: dataset loader, point-simulator
- Partial Cross-Entropy implementation
- Lightweight UNet and optional pretrained backbone stub
- Training loop, evaluation (mIoU), and experiment runner
- Example experiment cells: vary number of points and sampling strategy
- Save checkpoints, history, and produce simple plots

Instructions:
1) Place your dataset with structure:
   ./data/images/train/*.png
   ./data/masks/train/*.png
   ./data/images/val/*.png
   ./data/masks/val/*.png
   Masks must be single-channel with values 0..C-1 for classes.

2) Run: python partial_point_supervision_notebook.py --help

Notes:
- This script is runnable as a Python script and can be converted into cells for a Jupyter notebook.
- Internet access is not required by the script itself; download datasets manually.

"""

import os
import argparse
import random
from pathlib import Path
from typing import Tuple, List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# Utilities
# -------------------------
class RemoteSensingSegDataset(Dataset):
    """Simple dataset loader expecting matching filenames for images and masks."""
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(list(Path(images_dir).glob('*')))
        self.masks = sorted(list(Path(masks_dir).glob('*')))
        assert len(self.images) == len(self.masks), f"Counts don't match: {len(self.images)} vs {len(self.masks)}"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        img = np.array(img)
        mask = np.array(mask)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        mask = torch.from_numpy(mask).long()
        return img, mask, self.images[idx].name


def simulate_point_labels(mask: np.ndarray, num_points: int, strategy: str='random', seed: int=None) -> np.ndarray:
    """Simulate sparse labeling from a full mask. Unlabeled -> 255 (ignore_index).
    Strategies: 'random', 'class_balanced', 'edge_focused'."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    H, W = mask.shape
    annotated = np.ones_like(mask, dtype=np.uint8) * 255

    unique_classes = np.unique(mask)
    unique_classes = unique_classes[unique_classes != 255]

    if strategy == 'random':
        all_coords = [(i,j) for i in range(H) for j in range(W)]
        sampled = random.sample(all_coords, min(num_points, len(all_coords)))
        for (i,j) in sampled:
            annotated[i,j] = int(mask[i,j])
    elif strategy == 'class_balanced':
        classes = [int(c) for c in unique_classes]
        k = max(1, len(classes))
        pts_per_class = max(1, num_points // k)
        sampled = []
        for c in classes:
            ys, xs = np.where(mask == c)
            coords_c = list(zip(ys.tolist(), xs.tolist()))
            if len(coords_c) == 0:
                continue
            chosen = random.sample(coords_c, min(pts_per_class, len(coords_c)))
            sampled += chosen
        if len(sampled) < num_points:
            all_coords = [(i,j) for i in range(H) for j in range(W) if (i,j) not in sampled]
            more = random.sample(all_coords, min(num_points - len(sampled), len(all_coords)))
            sampled += more
        for (i,j) in sampled:
            annotated[i,j] = int(mask[i,j])
    elif strategy == 'edge_focused':
        try:
            import cv2
            edges = cv2.Canny(mask.astype('uint8')*50, 10, 100)
            ys, xs = np.where(edges > 0)
            coords_edge = list(zip(ys.tolist(), xs.tolist()))
            if len(coords_edge) == 0:
                return simulate_point_labels(mask, num_points, 'random', seed=seed)
            sampled = random.sample(coords_edge, min(num_points, len(coords_edge)))
            for (i,j) in sampled:
                annotated[i,j] = int(mask[i,j])
        except Exception:
            return simulate_point_labels(mask, num_points, 'random', seed=seed)
    else:
        raise ValueError('Unknown sampling strategy')

    return annotated

# -------------------------
# Partial Cross-Entropy
# -------------------------
class PartialCELoss(nn.Module):
    """CrossEntropyLoss that ignores unlabeled pixels set to 255."""
    def __init__(self, ignore_index=255, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        return self.ce(logits, target)

# -------------------------
# Lightweight UNet
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNetSimple(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f*2)
        self.enc3 = DoubleConv(f*2, f*4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(f*6, f*2)
        self.dec2 = DoubleConv(f*3, f)
        self.outc = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        u3 = self.up(e3)
        cat3 = torch.cat([u3, e2], dim=1)
        d3 = self.dec3(cat3)
        u2 = self.up(d3)
        cat2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(cat2)
        out = self.outc(d2)
        return out

# -------------------------
# Metrics
# -------------------------
def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    ious = []
    mask_valid = (gt != 255)
    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c = (gt == c)
        inter = np.logical_and(pred_c, gt_c) & mask_valid
        union = np.logical_or(pred_c, gt_c) & mask_valid
        if union.sum() == 0:
            ious.append(np.nan)
        else:
            ious.append(inter.sum() / float(union.sum()))
    ious = np.array(ious)
    return float(np.nanmean(ious))

# -------------------------
# Training and evaluation
# -------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device, num_classes):
    model.eval()
    iou_list = []
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks_np = masks.numpy()
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for p, g in zip(preds, masks_np):
                iou = compute_iou(p, g, num_classes)
                iou_list.append(iou)
    return float(np.nanmean(iou_list))

# -------------------------
# Experiment runner
# -------------------------

def run_experiment(images_dir, masks_dir, output_root='outputs', num_points=5, sampling='random',
                   epochs=10, batch_size=4, lr=1e-3, seed=42, num_classes=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train_images = os.path.join(images_dir, 'train')
    train_masks = os.path.join(masks_dir, 'train')
    val_images = os.path.join(images_dir, 'val')
    val_masks = os.path.join(masks_dir, 'val')

    os.makedirs(output_root, exist_ok=True)

    train_ds_full = RemoteSensingSegDataset(train_images, train_masks, transform=None)
    val_ds_full = RemoteSensingSegDataset(val_images, val_masks, transform=None)

    class PartialDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, num_points, sampling):
            self.base = base_ds
            self.num_points = num_points
            self.sampling = sampling
        def __len__(self):
            return len(self.base)
        def __getitem__(self, idx):
            img, mask, name = self.base[idx]
            mask_np = mask.numpy()
            annotated = simulate_point_labels(mask_np, self.num_points, self.sampling)
            if isinstance(img, np.ndarray):
                img_t = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            else:
                img_t = img
            annotated_t = torch.from_numpy(annotated).long()
            return img_t, annotated_t, name

    train_ds = PartialDataset(train_ds_full, num_points=num_points, sampling=sampling)
    val_ds = PartialDataset(val_ds_full, num_points=1000000, sampling='random')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = UNetSimple(in_channels=3, num_classes=num_classes, base_filters=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = PartialCELoss(ignore_index=255)

    history = {'train_loss':[], 'val_iou':[]}
    best_val = -1.0

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_iou = evaluate(model, val_loader, device, num_classes)
        history['train_loss'].append(train_loss)
        history['val_iou'].append(val_iou)
        print(f'Epoch {epoch:02d} | train_loss: {train_loss:.4f} | val_mIoU: {val_iou:.4f}')
        ckpt = os.path.join(output_root, f'model_epoch{epoch}.pth')
        torch.save({'epoch':epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt)
        if val_iou > best_val:
            best_val = val_iou
            torch.save(model.state_dict(), os.path.join(output_root, 'best_model.pth'))

    np.save(os.path.join(output_root, 'history.npy'), history)
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_iou'], label='val_iou')
    plt.legend()
    plt.title('Run history')
    plt.savefig(os.path.join(output_root, 'history.png'))
    print('Saved outputs to', output_root)

# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='./data/images')
    parser.add_argument('--masks_dir', type=str, default='./data/masks')
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--num_points', type=int, default=5)
    parser.add_argument('--sampling', type=str, default='random', choices=['random','class_balanced','edge_focused'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    run_experiment(images_dir=args.images_dir, masks_dir=args.masks_dir, output_root=args.output,
                   num_points=args.num_points, sampling=args.sampling, epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.lr, num_classes=args.num_classes)
