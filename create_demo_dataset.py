"""
Create a synthetic demo dataset for testing the partial-supervision segmentation script.
This creates simple images with geometric shapes and corresponding segmentation masks.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_segmentation_dataset(root='./data', num_train=20, num_val=10, img_size=128, num_classes=3):
    """
    Create a synthetic segmentation dataset with geometric shapes.
    Class 0: background (black)
    Class 1: circles (red)
    Class 2: rectangles (blue)
    """
    os.makedirs(f'{root}/images/train', exist_ok=True)
    os.makedirs(f'{root}/images/val', exist_ok=True)
    os.makedirs(f'{root}/masks/train', exist_ok=True)
    os.makedirs(f'{root}/masks/val', exist_ok=True)
    
    def create_sample(idx, is_train=True):
        # Create RGB image and grayscale mask
        img = Image.new('RGB', (img_size, img_size), color=(50, 50, 50))
        mask = Image.new('L', (img_size, img_size), color=0)
        
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Add random circles (class 1)
        num_circles = random.randint(2, 4)
        for _ in range(num_circles):
            r = random.randint(10, 20)
            cx = random.randint(r, img_size - r)
            cy = random.randint(r, img_size - r)
            bbox = [cx - r, cy - r, cx + r, cy + r]
            draw_img.ellipse(bbox, fill=(255, 100, 100), outline=(255, 50, 50))
            draw_mask.ellipse(bbox, fill=1, outline=1)
        
        # Add random rectangles (class 2)
        num_rects = random.randint(2, 4)
        for _ in range(num_rects):
            w = random.randint(15, 30)
            h = random.randint(15, 30)
            x = random.randint(0, img_size - w)
            y = random.randint(0, img_size - h)
            bbox = [x, y, x + w, y + h]
            draw_img.rectangle(bbox, fill=(100, 100, 255), outline=(50, 50, 255))
            draw_mask.rectangle(bbox, fill=2, outline=2)
        
        # Save
        split = 'train' if is_train else 'val'
        img.save(f'{root}/images/{split}/img_{idx:04d}.png')
        mask.save(f'{root}/masks/{split}/mask_{idx:04d}.png')
    
    print(f"Creating {num_train} training samples...")
    for i in range(num_train):
        random.seed(i)
        create_sample(i, is_train=True)
    
    print(f"Creating {num_val} validation samples...")
    for i in range(num_val):
        random.seed(1000 + i)
        create_sample(i, is_train=False)
    
    print(f"✓ Dataset created at '{root}/'")
    print(f"  - Training: {num_train} samples")
    print(f"  - Validation: {num_val} samples")
    print(f"  - Classes: {num_classes} (0=background, 1=circles, 2=rectangles)")

if __name__ == '__main__':
    create_synthetic_segmentation_dataset(
        root='./data',
        num_train=20,
        num_val=10,
        img_size=128,
        num_classes=3
    )

