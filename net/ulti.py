import torch.nn as nn
import torch.nn.functional as F

def init_weights(modules):
    pass

class ResidualBlock(nn.Module):#
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out



class ResidualBlock1(nn.Module):#
    def __init__(self,
                 in_channels, out_channels):
        super(ResidualBlock1, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),

        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


#
class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),

        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out

class Conv_Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(Conv_Block, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
        )

        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out