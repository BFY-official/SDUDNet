import torch
import torch.nn as nn
import torch.nn.functional as F



class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

    class BasicConv(nn.Module):
        def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                     bn=False, bias=False):
            super(BasicConv, self).__init__()
            self.out_channels = out_planes
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
            self.relu = nn.ReLU() if relu else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class _Residual_Block(nn.Module):
    def __init__(self, n_feat=64, reduction=8):
        super(_Residual_Block, self).__init__()
        self.SA = spatial_attn_layer() 
        self.CA = CALayer(n_feat, reduction)  
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))
        output = (self.conv2(output))
        output = torch.add(output, identity_data)
        output = self.relu(output)

        return output


class DNN(nn.Module):
    def __init__(self, stride=2):
        super(DNN, self).__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=3, ),
        )
        self.residual = self.make_layer(_Residual_Block, 6)
        self.conv_output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, ),
        )
        self.scale = nn.Parameter(torch.randn(3, 1, 1), requires_grad=True)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv_input(x)

        out = self.residual(out)

        out = self.conv_output(out)

        return out, x-out




if __name__ == "__main__":
    net = MDDN(gps=3, blocks=2)
    input = torch.rand(4,1,256,256)
    input = input.repeat(1,3,1,1)
    output = net(input)
    print(output.shape)