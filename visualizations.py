"""
Comprehensive visualization utilities for partial-supervision segmentation.

Features:
- Qualitative comparison: image, ground truth, point labels, prediction
- IoU per class visualization
- Confusion matrix
- Training curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path


def visualize_predictions(model, dataset, device, num_samples=5, output_path='visualizations.png', num_classes=3):
    """
    Create a comprehensive visualization grid showing:
    - Original image
    - Ground truth mask
    - Point labels (sparse supervision)
    - Model prediction
    """
    model.eval()
    
    # Color map for classes
    colors = plt.cm.get_cmap('tab10', num_classes)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(dataset))):
            img, sparse_mask, name = dataset[idx]
            
            # Get prediction
            img_batch = img.unsqueeze(0).to(device)
            logits = model(img_batch)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Convert tensors to numpy
            img_np = img.permute(1, 2, 0).cpu().numpy()
            sparse_mask_np = sparse_mask.cpu().numpy()
            
            # Get ground truth (for visualization only)
            # In practice, you'd have full mask for validation
            gt_mask = sparse_mask_np.copy()
            
            # Plot: Image
            axes[idx, 0].imshow(img_np)
            axes[idx, 0].set_title(f'Image: {name}')
            axes[idx, 0].axis('off')
            
            # Plot: Ground truth (when available)
            gt_colored = np.zeros((*gt_mask.shape, 3))
            for c in range(num_classes):
                mask_c = (gt_mask == c)
                gt_colored[mask_c] = colors(c)[:3]
            axes[idx, 1].imshow(img_np * 0.5 + gt_colored * 0.5)
            axes[idx, 1].set_title('Ground Truth Overlay')
            axes[idx, 1].axis('off')
            
            # Plot: Point labels (sparse supervision)
            axes[idx, 2].imshow(img_np)
            labeled_pixels = np.where(sparse_mask_np != 255)
            for y, x in zip(labeled_pixels[0], labeled_pixels[1]):
                label = sparse_mask_np[y, x]
                color = colors(label)
                circle = Circle((x, y), radius=2, color=color, alpha=0.8)
                axes[idx, 2].add_patch(circle)
            axes[idx, 2].set_title(f'Point Labels (n={len(labeled_pixels[0])})')
            axes[idx, 2].axis('off')
            
            # Plot: Prediction
            pred_colored = np.zeros((*pred.shape, 3))
            for c in range(num_classes):
                mask_c = (pred == c)
                pred_colored[mask_c] = colors(c)[:3]
            axes[idx, 3].imshow(img_np * 0.5 + pred_colored * 0.5)
            axes[idx, 3].set_title('Model Prediction')
            axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved visualization to {output_path}')
    plt.close()


def visualize_iou_per_class(iou_per_class, class_names=None, output_path='iou_per_class.png'):
    """
    Visualize IoU for each class as a bar chart.
    """
    num_classes = len(iou_per_class)
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_classes)
    bars = ax.bar(x, iou_per_class, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, iou_per_class)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('IoU', fontsize=12)
    ax.set_title('IoU per Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean IoU line
    mean_iou = np.nanmean(iou_per_class)
    ax.axhline(mean_iou, color='red', linestyle='--', linewidth=2, label=f'Mean IoU: {mean_iou:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved IoU per class to {output_path}')
    plt.close()


def compute_iou_per_class(model, loader, device, num_classes):
    """
    Compute IoU for each class separately.
    """
    model.eval()
    
    class_intersection = np.zeros(num_classes)
    class_union = np.zeros(num_classes)
    
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks_np = masks.numpy()
            
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for pred, gt in zip(preds, masks_np):
                valid_mask = (gt != 255)
                
                for c in range(num_classes):
                    pred_c = (pred == c) & valid_mask
                    gt_c = (gt == c) & valid_mask
                    
                    intersection = np.logical_and(pred_c, gt_c).sum()
                    union = np.logical_or(pred_c, gt_c).sum()
                    
                    class_intersection[c] += intersection
                    class_union[c] += union
    
    iou_per_class = np.zeros(num_classes)
    for c in range(num_classes):
        if class_union[c] > 0:
            iou_per_class[c] = class_intersection[c] / class_union[c]
        else:
            iou_per_class[c] = np.nan
    
    return iou_per_class


def plot_training_curves(history_path, output_path='training_curves.png'):
    """
    Plot training loss and validation IoU curves.
    """
    history = np.load(history_path, allow_pickle=True).item()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], marker='o', linewidth=2, markersize=6, color='steelblue')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Validation IoU
    axes[1].plot(epochs, history['val_iou'], marker='s', linewidth=2, markersize=6, color='forestgreen')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('mIoU', fontsize=12)
    axes[1].set_title('Validation mIoU', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved training curves to {output_path}')
    plt.close()


def create_confusion_matrix(model, loader, device, num_classes, class_names=None, output_path='confusion_matrix.png'):
    """
    Create and visualize confusion matrix for segmentation.
    """
    model.eval()
    
    if class_names is None:
        class_names = [f'C{i}' for i in range(num_classes)]
    
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks_np = masks.numpy()
            
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            for pred, gt in zip(preds, masks_np):
                valid_mask = (gt != 255)
                pred_valid = pred[valid_mask]
                gt_valid = gt[valid_mask]
                
                for c_gt in range(num_classes):
                    for c_pred in range(num_classes):
                        confusion[c_gt, c_pred] += ((gt_valid == c_gt) & (pred_valid == c_pred)).sum()
    
    # Normalize by row (true class)
    confusion_norm = confusion.astype(float)
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    confusion_norm = confusion_norm / row_sums
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Count', fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, f'{confusion_norm[i, j]:.2f}\n({confusion[i, j]})',
                         ha="center", va="center", color="black" if confusion_norm[i, j] < 0.5 else "white",
                         fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'✓ Saved confusion matrix to {output_path}')
    plt.close()


if __name__ == '__main__':
    print("Visualization utilities loaded.")
    print("Use these functions to visualize your segmentation results:")
    print("  - visualize_predictions(): Compare images, GT, point labels, and predictions")
    print("  - visualize_iou_per_class(): Bar chart of IoU for each class")
    print("  - plot_training_curves(): Training loss and validation IoU over epochs")
    print("  - create_confusion_matrix(): Pixel-wise confusion matrix")

