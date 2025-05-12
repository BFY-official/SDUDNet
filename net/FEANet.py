import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ShuffleAttention(nn.Module):
    def __init__(self, channel=32, reduction=16, G=1):
        super().__init__()
        self.G = G
        self.channel = channel  
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G)).to(device) 
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1)).to(device)
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1)).to(device)
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1)).to(device)
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1)).to(device)
        self.sigmoid = nn.Sigmoid() 

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)

        x_0, x_1 = x.chunk(2, dim=1)

        x_channel = self.avg_pool(x_0)  
        x_channel = self.cweight * x_channel + self.cbias 
        x_channel = x_0 * self.sigmoid(x_channel) 

        x_spatial = self.gn(x_1) 
        x_spatial = self.sweight * x_spatial + self.sbias 
        x_spatial = x_1 * self.sigmoid(x_spatial)

        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w) 

        out = self.channel_shuffle(out, 2)
        return out


class P_Conv(nn.Module):
    def __init__(self):
        super(P_Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        return x


class B_Conv(nn.Module):
    def __init__(self):
        super(B_Conv, self).__init__()
        self.conv1 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(4, 1, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        return x


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
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

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7 
        self.compress = ZPool()  
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False) 

    def forward(self, x):
        x_compress = self.compress(x)  
        x_out = self.conv(x_compress) 
        scale = torch.sigmoid_(x_out)
        return x * scale 

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.p_conv = P_Conv()
        self.b_conv = B_Conv()
        self.cw = AttentionGate()  
        self.hc = AttentionGate() 
        self.no_spatial = no_spatial 
        if not no_spatial:
            self.hw = AttentionGate() 

    def forward(self, x):
        input = x
        x = self.p_conv(x)

        x_perm1 = x.permute(0, 2, 1, 3).contiguous() 
        # b, c1, h, w = x_perm1.shape
        # SE1 = ShuffleAttention(c1)
        x_out1 = self.cw(x_perm1)
        # x_out1 = SE1(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous() 
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        # b, c2, h, w = x_perm2.shape
        # SE2 = ShuffleAttention(c2)
        x_out2 = self.hc(x_perm2)
        # x_out2 = SE2(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            # b, c3, h, w = x.shape
            # SE3 = ShuffleAttention(c3)
            x_out = self.hw(x)  
            # x_out = SE3(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21) 
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)  
        x_out = self.b_conv(x_out)
        return torch.cat([x_out, input], dim=1)



if __name__ == '__main__':
    input = torch.randn(16, 1, 256, 128)
    triplet = TripletAttention() 
    output = triplet(input) 
    print(output.shape) 
    #
    # p_conv = P_Conv()
    # z = p_conv(input)
    #
    # b_conv = B_Conv()
    #
    # # print(z.shape)
    # output = triplet(z)
    #
    # output = b_conv(output)
    # print(output.shape) 
