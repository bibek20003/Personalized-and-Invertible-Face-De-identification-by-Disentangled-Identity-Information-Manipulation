import torch
import torch.nn as nn
import torch.nn.functional as F

class AADResBlock(nn.Module):
    def __init__(self, in_channels, attr_channels, id_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, z_attr, z_id):
        out = self.norm(x)
        out = self.conv(out)
        return out + x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_block = AADResBlock(512, 512, 512)

        # Upsampling blocks to go from 16×16 → 32×32 → 64×64 → 128×128 → 256×256
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU())
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU())
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())

        self.final = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, z_attr, z_id):
        x = self.init_block(z_attr, z_attr, z_id)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return torch.sigmoid(self.final(x))  # output size: (B, 3, 256, 256)
