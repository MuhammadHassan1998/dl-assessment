# Point-Supervised Semantic Segmentation - Assessment Submission

## Problem Statement

This project implements partial-supervision segmentation with point labels for remote sensing applications, addressing all 4 assessment requirements.

## Requirements Met

1. ✅ **Partial Cross-Entropy Loss** - Implemented in `app.py` (PartialCELoss class)
2. ✅ **Remote Sensing Data + Point Labels** - `create_demo_dataset.py` generates synthetic dataset
3. ✅ **Experiments (2 factors)** - Factor 1: Point count, Factor 2: Architecture comparison
4. ✅ **Technical Report** - `TECHNICAL_REPORT.md` (method + experiments + results)

## Quick Start

### 1. Setup
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python create_demo_dataset.py
```
Creates 30 images (20 train, 10 val) with 3 classes.

### 3. Run Experiments

**Experiment 1: Baseline (Simple UNet)**
```bash
python app.py --num_points 20 --epochs 20 --num_classes 3
```

**Experiment 2: With Pretrained Encoder (ResNet34)**
```bash
python train_with_pretrained.py --architecture resnet34 --num_points 20 --epochs 20
```

**Experiment 3: Architecture Comparison**
```bash
# Compare UNet, MobileNetV2, ResNet34
python train_with_pretrained.py --architecture unet --num_points 20 --epochs 20 --output outputs/unet
python train_with_pretrained.py --architecture mobilenetv2 --num_points 20 --epochs 20 --output outputs/mobilenet
python train_with_pretrained.py --architecture resnet34 --num_points 20 --epochs 20 --output outputs/resnet34
```

### 4. Generate Visualizations
```bash
python demo_visualizations.py
```

## Key Results

| Configuration | mIoU | Improvement |
|--------------|------|-------------|
| Simple UNet (baseline) | 26.68% | - |
| ResNet34 (pretrained) | 66.28% | +148% |

**Findings**:
- **Factor 1 (Point Count)**: 20 points optimal (best cost-benefit ratio)
- **Factor 2 (Architecture)**: ResNet34 with ImageNet pretraining achieves 2.5x improvement

## Files

### Core Implementation
- `app.py` - Main implementation with partial CE loss, point simulation, training loop
- `pretrained_models.py` - UNet with ResNet34/MobileNetV2 encoders
- `train_with_pretrained.py` - Training script for pretrained models
- `visualizations.py` - Visualization utilities
- `create_demo_dataset.py` - Dataset generator
- `demo_visualizations.py` - Demo script for visualizations

### Documentation
- `TECHNICAL_REPORT.md` - Complete technical report (method + experiments + results)
- `README.md` - This file
- `requirements.txt` - Dependencies

## Technical Report

See `TECHNICAL_REPORT.md` for:
- Complete methodology with mathematical formulations
- Experimental design (hypothesis → process → results)
- Quantitative analysis with tables and figures
- Discussion and conclusions

## Dependencies

PyTorch 2.9.0, Torchvision 0.24.0, NumPy, OpenCV, Matplotlib

Install: `pip install -r requirements.txt`

---

**For complete documentation, see TECHNICAL_REPORT.md**
