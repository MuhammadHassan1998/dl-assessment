"""
UNet with pretrained encoders (ResNet34, MobileNetV2) for improved performance.
Uses ImageNet-pretrained weights for the encoder backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class UNetResNet34(nn.Module):
    """
    UNet with ResNet34 pretrained encoder.
    Architecture follows a standard encoder-decoder structure with skip connections.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pretrained ResNet34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Encoder (ResNet34 layers)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.encoder2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels
        self.encoder5 = resnet.layer4  # 512 channels
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder5 = self._make_decoder_block(512, 256)
        
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_block(256, 128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_block(128, 64)
        
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_block(128, 64)
        
        # Final classifier
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with conv + bn + relu."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)      # 64, H/2, W/2
        e2 = self.encoder2(e1)     # 64, H/4, W/4
        e3 = self.encoder3(e2)     # 128, H/8, W/8
        e4 = self.encoder4(e3)     # 256, H/16, W/16
        e5 = self.encoder5(e4)     # 512, H/32, W/32
        
        # Decoder with skip connections
        d5 = self.upconv5(e5)                      # 256, H/16, W/16
        d5 = torch.cat([d5, e4], dim=1)            # 512, H/16, W/16
        d5 = self.decoder5(d5)                     # 256, H/16, W/16
        
        d4 = self.upconv4(d5)                      # 128, H/8, W/8
        d4 = torch.cat([d4, e3], dim=1)            # 256, H/8, W/8
        d4 = self.decoder4(d4)                     # 128, H/8, W/8
        
        d3 = self.upconv3(d4)                      # 64, H/4, W/4
        d3 = torch.cat([d3, e2], dim=1)            # 128, H/4, W/4
        d3 = self.decoder3(d3)                     # 64, H/4, W/4
        
        d2 = self.upconv2(d3)                      # 64, H/2, W/2
        d2 = torch.cat([d2, e1], dim=1)            # 128, H/2, W/2
        d2 = self.decoder2(d2)                     # 64, H/2, W/2
        
        # Upsample to original resolution
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final classification
        out = self.final(d2)
        
        return out


class UNetMobileNetV2(nn.Module):
    """
    UNet with MobileNetV2 pretrained encoder.
    Lightweight architecture suitable for resource-constrained scenarios.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Encoder (MobileNetV2 features)
        features = mobilenet.features
        self.encoder1 = features[0:2]   # 16 channels, stride 2
        self.encoder2 = features[2:4]   # 24 channels, stride 4
        self.encoder3 = features[4:7]   # 32 channels, stride 8
        self.encoder4 = features[7:14]  # 96 channels, stride 16
        self.encoder5 = features[14:]   # 1280 channels, stride 32
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(1280, 96, kernel_size=2, stride=2)
        self.decoder5 = self._make_decoder_block(192, 96)
        
        self.upconv4 = nn.ConvTranspose2d(96, 32, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_block(64, 32)
        
        self.upconv3 = nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_block(48, 24)
        
        self.upconv2 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_block(32, 16)
        
        # Final classifier
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a decoder block with depthwise separable convolutions (MobileNet style)."""
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)      # 16, H/2, W/2
        e2 = self.encoder2(e1)     # 24, H/4, W/4
        e3 = self.encoder3(e2)     # 32, H/8, W/8
        e4 = self.encoder4(e3)     # 96, H/16, W/16
        e5 = self.encoder5(e4)     # 1280, H/32, W/32
        
        # Decoder with skip connections
        d5 = self.upconv5(e5)                      # 96, H/16, W/16
        d5 = torch.cat([d5, e4], dim=1)            # 192, H/16, W/16
        d5 = self.decoder5(d5)                     # 96, H/16, W/16
        
        d4 = self.upconv4(d5)                      # 32, H/8, W/8
        d4 = torch.cat([d4, e3], dim=1)            # 64, H/8, W/8
        d4 = self.decoder4(d4)                     # 32, H/8, W/8
        
        d3 = self.upconv3(d4)                      # 24, H/4, W/4
        d3 = torch.cat([d3, e2], dim=1)            # 48, H/4, W/4
        d3 = self.decoder3(d3)                     # 24, H/4, W/4
        
        d2 = self.upconv2(d3)                      # 16, H/2, W/2
        d2 = torch.cat([d2, e1], dim=1)            # 32, H/2, W/2
        d2 = self.decoder2(d2)                     # 16, H/2, W/2
        
        # Upsample to original resolution
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final classification
        out = self.final(d2)
        
        return out


def get_model(architecture='unet', num_classes=2, pretrained=True):
    """
    Factory function to get the desired model architecture.
    
    Args:
        architecture: One of ['unet', 'resnet34', 'mobilenetv2']
        num_classes: Number of segmentation classes
        pretrained: Whether to use ImageNet pretrained weights (for resnet34/mobilenetv2)
    
    Returns:
        PyTorch model
    """
    if architecture == 'resnet34':
        return UNetResNet34(num_classes=num_classes, pretrained=pretrained)
    elif architecture == 'mobilenetv2':
        return UNetMobileNetV2(num_classes=num_classes, pretrained=pretrained)
    elif architecture == 'unet':
        # Import from app.py
        from app import UNetSimple
        return UNetSimple(in_channels=3, num_classes=num_classes, base_filters=32)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def compare_model_sizes():
    """Compare model sizes and parameter counts."""
    import torch
    
    models_dict = {
        'UNet (Simple)': get_model('unet', num_classes=2, pretrained=False),
        'UNet-ResNet34': get_model('resnet34', num_classes=2, pretrained=False),
        'UNet-MobileNetV2': get_model('mobilenetv2', num_classes=2, pretrained=False),
    }
    
    print("Model Comparison:")
    print("-" * 60)
    print(f"{'Model':<25} {'Parameters':>15} {'Size (MB)':>15}")
    print("-" * 60)
    
    for name, model in models_dict.items():
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = num_params * 4 / (1024 ** 2)  # Assuming float32
        print(f"{name:<25} {num_params:>15,} {size_mb:>15.2f}")
    
    print("-" * 60)


if __name__ == '__main__':
    print("Pretrained Model Architectures for Segmentation\n")
    
    # Compare model sizes
    compare_model_sizes()
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 128, 128)
    
    for arch in ['unet', 'resnet34', 'mobilenetv2']:
        model = get_model(arch, num_classes=3, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"  {arch:15} -> Input: {tuple(x.shape)}, Output: {tuple(out.shape)} ✓")

