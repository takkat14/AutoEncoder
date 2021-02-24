class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel-1)//2, bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return self.act(self.norm(self.conv(x)))

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            Block(1, 16, 3, stride=2),
            Block(16, 32, 3, stride=2),
            Block(32, 32, 3, stride=2),
            Block(32, 32, 3, stride=2),
            Block(32, 32, 3, stride=1).conv,
        )

        self.decoder = nn.Sequential(
            Block(32, 32, 3, upsample=True),
            Block(32, 32, 3, upsample=True),
            Block(32, 32, 3, upsample=True),
            Block(32, 16, 3, upsample=True),
            Block(16, 1, 3).conv,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

    def get_latent_features(self, x):
        return self.encoder(x)
