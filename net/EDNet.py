import functools
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from net import FEANet



class UNetEncoder(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(UNetEncoder, self).__init__()

        self.initial_conv = nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)
        self.initial_relu = nn.LeakyReLU(0.2, False)

        layers = []
        nf_mult_prev = 1
        for n in range(1, n_layers + 1):
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]
            nf_mult_prev = nf_mult

        layers += [
            nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.initial_conv(input)
        x = self.initial_relu(x)
        x = self.layers(x)
        return x



class UNetDecoder(nn.Module):
    def __init__(self, output_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(UNetDecoder, self).__init__()

        layers = []
        nf_mult = min(2 ** n_layers, 8)
        for n in range(n_layers, 0, -1):
            layers += [
                nn.ConvTranspose2d(ndf * nf_mult, ndf * nf_mult // 2, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult // 2),
                nn.ReLU(True)
            ]
            nf_mult //= 2

        layers += [
            nn.ConvTranspose2d(ndf, output_nc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, False)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x


class UNet(nn.Module):
    def __init__(self, input_nc=2, output_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(UNet, self).__init__()

        sequence = [
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)] 
        self.model = nn.Sequential(*sequence)
        self.encoder = UNetEncoder(input_nc, ndf, n_layers, norm_layer)
        self.decoder = UNetDecoder(output_nc, ndf, n_layers, norm_layer)
        self.fea = FEANet.TripletAttention()
    def forward(self, input):
        input = self.fea(input)
        enc = self.encoder(input)
        dec = self.decoder(enc)
        
        return torch.sigmoid(self.model(enc)), torch.sigmoid(dec)



if __name__ == '__main__':
    in_channels = 1
    H, W = 256, 256
    x = torch.randn(size=(16, in_channels, H, W))
    disc = UNet()
    enc, dec = disc(x)

    loss = nn.MSELoss()
    loss = loss(enc, torch.ones_like(enc))

    print(enc.shape)
    print(dec.shape)





