import torch
import torch.nn as nn
import torch.nn.functional as F

class DDRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DDRB, self).__init__()
        self.dense1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.dense2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dense3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv1x1 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.residual = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.dense1(x))
        out = F.relu(self.dense2(out))
        out = F.relu(self.dense3(out))
        out = self.conv1x1(out)
        out += self.residual(x)
        return out

class ReverseAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReverseAttention, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(1 - x)

class BTISNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BTISNet, self).__init__()
        self.encoder1 = DDRB(in_channels, 64)
        self.encoder2 = DDRB(64, 128)
        self.encoder3 = DDRB(128, 256)
        self.encoder4 = DDRB(256, 512)

        self.decoder1 = DDRB(512, 256)
        self.decoder2 = DDRB(256, 128)
        self.decoder3 = DDRB(128, 64)
        self.decoder4 = DDRB(64, out_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

        self.reverse_attention = ReverseAttention(out_channels, out_channels)

    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.downsample(e1))
        e3 = self.encoder3(self.downsample(e2))
        e4 = self.encoder4(self.downsample(e3))

        # Decoding path
        d1 = self.decoder1(self.upsample(e4) + e3)
        d2 = self.decoder2(self.upsample(d1) + e2)
        d3 = self.decoder3(self.upsample(d2) + e1)
        d4 = self.decoder4(self.upsample(d3))

        # Reverse Attention
        out = torch.sigmoid(d4)
        reverse_attention1 = self.reverse_attention(out)
        reverse_attention2 = self.reverse_attention(reverse_attention1)
        reverse_attention3 = self.reverse_attention(reverse_attention2)

        return out, reverse_attention1, reverse_attention2, reverse_attention3
