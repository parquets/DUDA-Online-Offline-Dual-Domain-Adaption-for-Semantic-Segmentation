import torch
import torch.nn as nn

class FCDiscriminator(nn.Module):
    def __init__(self, in_channels, ndf = 64):
        super(FCDiscriminator, self).__init__()
        self.pool_bin = (2, 4, 8, 16)
        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):
        if isinstance(img, list) or isinstance(img, tuple):
            img = img[-1]
        x = self.conv1(img)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
