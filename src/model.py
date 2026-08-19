import torch
import torch.nn as nn

class DoubleConv3D(nn.Module):
    """
    A block containing:
    (Conv3D -> InstanceNorm3D -> LeakyReLU) x 2
    
    We use InstanceNorm3D instead of BatchNorm3D because batch sizes in 3D segmentation 
    are extremely small (typically 1 or 2). Batch Normalization fails at small batch 
    sizes because it cannot compute reliable batch statistics. Instance Normalization 
    normalizes each sample independently.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net architecture for brain tumor segmentation.
    This model has an encoder-decoder structure with skip connections:
    - Input: 4-channel image (FLAIR, T1, T1CE, T2) of shape (Batch, 4, 128, 128, 128)
    - Output: 3-channel logits (WT, TC, ET) of shape (Batch, 3, 128, 128, 128)
    
    Channels: 4 -> 16 -> 32 -> 64 -> 128 (bottleneck) -> 64 -> 32 -> 16 -> 3.
    This is smaller than standard U-Net to be memory-efficient on standard GPUs / M1 Mac.
    """
    def __init__(self, in_channels=4, out_channels=3, base_filters=16):
        super().__init__()

        # Encoder (Downsampling path)
        self.enc1 = DoubleConv3D(in_channels, base_filters)
        self.pool1 = nn.Conv3d(base_filters, base_filters, kernel_size=3, stride=2, padding=1, bias=False)  # Halves dimensions: 128 -> 64

        self.enc2 = DoubleConv3D(base_filters, base_filters * 2)
        self.pool2 = nn.Conv3d(base_filters * 2, base_filters * 2, kernel_size=3, stride=2, padding=1, bias=False)  # Halves dimensions: 64 -> 32

        self.enc3 = DoubleConv3D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.Conv3d(base_filters * 4, base_filters * 4, kernel_size=3, stride=2, padding=1, bias=False)  # Halves dimensions: 32 -> 16

        # Bottleneck (Bridge)
        self.bottleneck = DoubleConv3D(base_filters * 4, base_filters * 8)

        # Decoder (Upsampling path)
        # We use trilinear upsampling which does not require learnable parameters, reducing memory
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec3 = DoubleConv3D(base_filters * 8 + base_filters * 4, base_filters * 4)  # 128 + 64 = 192 in

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec2 = DoubleConv3D(base_filters * 4 + base_filters * 2, base_filters * 2)  # 64 + 32 = 96 in

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec1 = DoubleConv3D(base_filters * 2 + base_filters, base_filters)         # 32 + 16 = 48 in

        # Final 1x1x1 convolution to get the number of output classes
        # Note: We output logits. Sigmoid activation is applied inside the loss function 
        # and during inference for numerical stability.
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(pool3)

        # Decoder
        up3 = self.up3(bottleneck)
        concat3 = torch.cat([up3, enc3], dim=1)  # Concatenate along channel dimension
        dec3 = self.dec3(concat3)

        up2 = self.up2(dec3)
        concat2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(concat2)

        up1 = self.up1(dec2)
        concat1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(concat1)

        logits = self.final_conv(dec1)
        return logits

if __name__ == "__main__":
    # Test network input/output shapes
    device = torch.device("cpu")
    model = UNet3D().to(device)
    dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)
    print("Testing 3D U-Net model forward pass...")
    print("Input shape:", dummy_input.shape)
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape (logits):", output.shape)
    assert output.shape == (1, 3, 128, 128, 128), "Error: Shape mismatch in output!"
    print("Shape test passed successfully!")
