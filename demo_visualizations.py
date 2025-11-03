"""
Demo script showing how to use the visualization utilities.
Run this after training a model with app.py
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Import from app.py
from app import RemoteSensingSegDataset, UNetSimple, simulate_point_labels
from visualizations import (
    visualize_predictions, 
    visualize_iou_per_class, 
    plot_training_curves,
    create_confusion_matrix,
    compute_iou_per_class
)


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


def main():
    # Configuration
    model_path = './outputs/test_run/best_model.pth'
    history_path = './outputs/test_run/history.npy'
    output_dir = './outputs/test_run/visualizations'
    
    data_root = './data'
    num_classes = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("VISUALIZATION DEMO FOR PARTIAL-SUPERVISION SEGMENTATION")
    print("=" * 60)
    
    # Load model
    print("\n[1/6] Loading model...")
    model = UNetSimple(in_channels=3, num_classes=num_classes, base_filters=32).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠ Model not found at {model_path}. Using untrained model.")
    
    # Load validation dataset
    print("\n[2/6] Loading validation dataset...")
    val_images = os.path.join(data_root, 'images/val')
    val_masks = os.path.join(data_root, 'masks/val')
    val_ds_full = RemoteSensingSegDataset(val_images, val_masks, transform=None)
    val_ds = PartialDataset(val_ds_full, num_points=1000000, sampling='random')
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"✓ Loaded {len(val_ds)} validation samples")
    
    # 1. Visualize predictions
    print("\n[3/6] Creating prediction visualizations...")
    visualize_predictions(
        model, val_ds, device, 
        num_samples=min(5, len(val_ds)),
        output_path=os.path.join(output_dir, 'predictions.png'),
        num_classes=num_classes
    )
    
    # 2. Compute and visualize IoU per class
    print("\n[4/6] Computing IoU per class...")
    iou_per_class = compute_iou_per_class(model, val_loader, device, num_classes)
    class_names = ['Background', 'Circles', 'Rectangles']
    print(f"IoU per class:")
    for i, (name, iou) in enumerate(zip(class_names, iou_per_class)):
        print(f"  {name}: {iou:.4f}")
    print(f"  Mean IoU: {iou_per_class.mean():.4f}")
    
    visualize_iou_per_class(
        iou_per_class, 
        class_names=class_names,
        output_path=os.path.join(output_dir, 'iou_per_class.png')
    )
    
    # 3. Plot training curves
    print("\n[5/6] Plotting training curves...")
    if os.path.exists(history_path):
        plot_training_curves(
            history_path,
            output_path=os.path.join(output_dir, 'training_curves.png')
        )
    else:
        print(f"⚠ History not found at {history_path}")
    
    # 4. Create confusion matrix
    print("\n[6/6] Creating confusion matrix...")
    create_confusion_matrix(
        model, val_loader, device, num_classes,
        class_names=class_names,
        output_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    print("\n" + "=" * 60)
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print("=" * 60)
    print("\nGenerated files:")
    print(f"  1. predictions.png - Qualitative comparison")
    print(f"  2. iou_per_class.png - IoU bar chart")
    print(f"  3. training_curves.png - Loss and mIoU over epochs")
    print(f"  4. confusion_matrix.png - Pixel-wise confusion matrix")


if __name__ == '__main__':
    main()

