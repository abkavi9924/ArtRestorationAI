import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.checkpoint import checkpoint


class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 60, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize weights for identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 60 * 60) 
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

# U-Net Restoration Model
class RestorationUNet(nn.Module):
    def __init__(self):
        super(RestorationUNet, self).__init__()
        self.stn = SpatialTransformer()

        # Color correction module
        self.color_corrector = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

        # U-Net for restoration
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.out = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.stn(x)
        x = self.color_corrector(x)
        e1 = checkpoint(self.enc1, x)
        e2 = checkpoint(self.enc2, F.max_pool2d(e1, 2))
        e3 = checkpoint(self.enc3, F.max_pool2d(e2, 2))
        e4 = checkpoint(self.enc4, F.max_pool2d(e3, 2))
        b = checkpoint(self.bottleneck, F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([F.interpolate(b, scale_factor=2), e4], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))
        return self.out(d1)


# Perceptual loss using VGG16
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.device = device
        self.vgg.to(self.device)

    def forward(self, output, target):
        output = output.to(self.device)
        target = target.to(self.device)
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        return nn.MSELoss()(output_features, target_features)

