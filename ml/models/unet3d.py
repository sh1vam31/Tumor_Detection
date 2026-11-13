import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    """(3D Convolution -> InstanceNorm -> LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """
    3D U-Net architecture for BraTS Brain Tumor Segmentation.
    Input shape:  [Batch, InChannels (4: T1, T1ce, T2, FLAIR), Depth, Height, Width]
    Output shape: [Batch, OutClasses (3: Whole Tumor, Tumor Core, Enhancing Tumor), Depth, Height, Width]
    """
    def __init__(self, in_channels=4, out_channels=3, init_features=16):
        super(UNet3D, self).__init__()
        features = init_features

        # Encoder (Contracting Path)
        self.encoder1 = DoubleConv3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(features * 4, features * 8)

        # Decoder (Expanding Path)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(features * 2 * 2, features * 2)

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(features * 2, features)

        # Final Segmentation Head
        self.final_conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder with Skip Connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output logits
        return self.final_conv(dec1)

if __name__ == "__main__":
    model = UNet3D(in_channels=4, out_channels=3, init_features=16)
    dummy_input = torch.randn(1, 4, 32, 64, 64)
    output = model(dummy_input)
    print("3D U-Net Model initialized successfully!")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
