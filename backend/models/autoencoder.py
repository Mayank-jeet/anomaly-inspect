import torch
import torch.nn as nn


def _down_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up_block(in_ch, out_ch, final=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
    ]
    if final:
        layers.append(nn.Sigmoid())
    else:
        layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class ConvAE(nn.Module):
    def __init__(self, base_ch: int = 32, latent_ch: int = 256):
        super().__init__()
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.encoder = nn.Sequential(
            _down_block(3, base_ch),          # 112, base_ch
            _down_block(base_ch, base_ch*2),  # 56,  2*base_ch
            _down_block(base_ch*2, base_ch*4),# 28,  4*base_ch
            _down_block(base_ch*4, base_ch*8),# 14,  8*base_ch
            _down_block(base_ch*8, latent_ch),# 7,   latent_ch
        )
        self.decoder = nn.Sequential(
            _up_block(latent_ch, base_ch*8),  # 14
            _up_block(base_ch*8, base_ch*4),  # 28
            _up_block(base_ch*4, base_ch*2),  # 56
            _up_block(base_ch*2, base_ch),    # 112
            _up_block(base_ch, 3, final=True),# 224
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


if __name__ == "__main__":
    m = ConvAE()
    x = torch.zeros(2, 3, 224, 224)
    out = m(x)
    assert out.shape == x.shape, (out.shape, x.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print("output shape:", tuple(out.shape))
    print("params:", n_params)