"""Frozen ResNet-18 feature extractor for PatchCore. layer2+layer3, upsampled, concatenated, locally averaged."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PatchFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        # Keep only the layers we need, in order: stem -> layer1 -> layer2 -> layer3
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)                      # [B,128,28,28]
        f3 = self.layer3(f2)                      # [B,256,14,14]
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f2, f3_up], dim=1)       # [B,384,28,28]
        feat = self.pool(feat)                     # local neighborhood averaging, same shape
        return feat

    @torch.no_grad()
    def extract_patches(self, x):
        """[B,384,28,28] -> [B*28*28, 384] patch matrix."""
        feat = self.forward(x)
        B, C, H, W = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return patches


if __name__ == "__main__":
    model = PatchFeatureExtractor().eval()
    x = torch.zeros(1, 3, 224, 224)
    feat = model(x)
    patches = model.extract_patches(x)
    print("feat shape:", tuple(feat.shape))
    print("patches shape:", tuple(patches.shape))

    # determinism check — run twice, confirm identical
    feat2 = model(x)
    print("deterministic:", torch.equal(feat, feat2))