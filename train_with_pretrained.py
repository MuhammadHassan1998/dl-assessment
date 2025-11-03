"""
Enhanced training script with pretrained encoder support.
Supports: UNet (simple), ResNet34, MobileNetV2
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app import RemoteSensingSegDataset, PartialCELoss, simulate_point_labels
from pretrained_models import get_model


class PartialDataset(torch.utils.data.Dataset):
    """Dataset wrapper that simulates point labels."""
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
        if isinstance(img, torch.Tensor):
            img_t = img
        else:
            img_t = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        annotated_t = torch.from_numpy(annotated).long()
        return img_t, annotated_t, name


def compute_iou(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> float:
    """Compute mean IoU."""
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
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
    """Evaluate model on validation set."""
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


def run_experiment(args):
    """Main training experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("TRAINING WITH PRETRAINED ENCODERS")
    print("=" * 70)
    print(f"Architecture: {args.architecture}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Device: {device}")
    print(f"Num classes: {args.num_classes}")
    print(f"Point supervision: {args.num_points} points ({args.sampling})")
    print("=" * 70)
    
    # Prepare data paths
    train_images = os.path.join(args.images_dir, 'train')
    train_masks = os.path.join(args.masks_dir, 'train')
    val_images = os.path.join(args.images_dir, 'val')
    val_masks = os.path.join(args.masks_dir, 'val')
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load datasets
    print("\n[1/4] Loading datasets...")
    train_ds_full = RemoteSensingSegDataset(train_images, train_masks, transform=None)
    val_ds_full = RemoteSensingSegDataset(val_images, val_masks, transform=None)
    
    train_ds = PartialDataset(train_ds_full, num_points=args.num_points, sampling=args.sampling)
    val_ds = PartialDataset(val_ds_full, num_points=1000000, sampling='random')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"✓ Training samples: {len(train_ds)}")
    print(f"✓ Validation samples: {len(val_ds)}")
    
    # Create model
    print(f"\n[2/4] Creating model: {args.architecture}...")
    model = get_model(args.architecture, num_classes=args.num_classes, pretrained=args.pretrained).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {num_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = PartialCELoss(ignore_index=255)
    
    # Training loop
    print(f"\n[3/4] Training for {args.epochs} epochs...")
    history = {'train_loss': [], 'val_iou': []}
    best_val = -1.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_iou = evaluate(model, val_loader, device, args.num_classes)
        history['train_loss'].append(train_loss)
        history['val_iou'].append(val_iou)
        
        print(f'Epoch {epoch:02d}/{args.epochs} | Loss: {train_loss:.4f} | Val mIoU: {val_iou:.4f}')
        
        # Save checkpoint
        ckpt = os.path.join(args.output, f'model_epoch{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'architecture': args.architecture,
            'num_classes': args.num_classes
        }, ckpt)
        
        # Save best model
        if val_iou > best_val:
            best_val = val_iou
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print(f"  → Best model saved (mIoU: {best_val:.4f})")
    
    # Save history and plot
    print(f"\n[4/4] Saving results...")
    np.save(os.path.join(args.output, 'history.npy'), history)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, args.epochs + 1)
    
    axes[0].plot(epochs_range, history['train_loss'], marker='o', linewidth=2, color='steelblue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(epochs_range, history['val_iou'], marker='s', linewidth=2, color='forestgreen')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Validation mIoU')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'history.png'), dpi=150)
    
    print("=" * 70)
    print(f"✓ TRAINING COMPLETE")
    print(f"  Best validation mIoU: {best_val:.4f}")
    print(f"  Outputs saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation with pretrained encoders')
    
    # Data
    parser.add_argument('--images_dir', type=str, default='./data/images')
    parser.add_argument('--masks_dir', type=str, default='./data/masks')
    parser.add_argument('--output', type=str, default='./outputs/pretrained')
    
    # Model
    parser.add_argument('--architecture', type=str, default='resnet34', 
                       choices=['unet', 'resnet34', 'mobilenetv2'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use ImageNet pretrained weights')
    parser.add_argument('--num_classes', type=int, default=3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Supervision
    parser.add_argument('--num_points', type=int, default=10)
    parser.add_argument('--sampling', type=str, default='class_balanced',
                       choices=['random', 'class_balanced', 'edge_focused'])
    
    args = parser.parse_args()
    run_experiment(args)

