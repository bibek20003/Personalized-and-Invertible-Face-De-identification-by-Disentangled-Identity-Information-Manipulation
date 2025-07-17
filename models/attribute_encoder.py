# models/attribute_encoder.py

import torch
import torch.nn as nn

class AttributeEncoder(nn.Module):
    def __init__(self):
        super(AttributeEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)
