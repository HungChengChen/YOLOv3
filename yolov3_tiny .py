import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module


class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(CBL, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)
        return out


class MCBL(nn.Module):
    def __init__(self, in_channels, out_channels, dowmsample=True):
        super(MCBL, self).__init__()
        #padding = (kernel_size - 1) // 2
        if dowmsample == False:
            self.max = nn.MaxPool2d(kernel_size=2, stride=1)
        else:
            self.max = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.max(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.lrelu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return x


class BackoneNet(nn.Module):
    def __init__(self):
        super(BackoneNet, self).__init__()
        self.conv = CBL(3, 16)
        self.layer1 = MCBL(16, 32)
        self.layer2 = MCBL(32, 64)
        self.layer3 = MCBL(64, 128)
        self.layer4 = MCBL(128, 256)
        self.layer5 = MCBL(256, 512)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.layer6 = MCBL(512, 1024, dowmsample=False)
        self.layer7 = CBL(1024, 256)

    def forward(self, x):
        tmp = self.conv(x)
        tmp = self.layer1(tmp)
        tmp = self.layer2(tmp)
        out3 = self.layer3(tmp)
        out2 = self.layer4(out3)
        tmp = self.layer5(out2)
        tmp = self.pad(tmp)
        tmp = self.layer6(tmp)
        out1 = self.layer7(tmp)
        return out1, out2, out3


class FPN(nn.Module):
    def __init__(self, num_class, num_anchor=3):
        super(FPN, self).__init__()
        self.num_class = num_class
        self.model = BackoneNet()
        self.predict_1 = nn.Sequential(CBL(256, 512, 3), nn.Conv2d(512, (4 + 1 + self.num_class) * num_anchor, 1))

        self.CBL_Up2D_1 = nn.Sequential(CBL(256, 128, 1), Interpolate(scale_factor=2, mode="nearest"))
        self.predict_2 = nn.Sequential(CBL(384, 256, 3), nn.Conv2d(256, (4 + 1 + self.num_class) * num_anchor, 1))

    def forward(self, tmp1, tmp2, tmp3):
        output = []
        out1 = self.predict_1(tmp1)
        tmp = torch.cat((self.CBL_Up2D_1(tmp1), tmp2), 1)
        out2 = self.predict_2(tmp)
        output = [out1, out2]
        return output

    def _make_CBLx5_(self, in_channels, out_channels):
        assert in_channels % 2 == 0
        CBLx5 = nn.Sequential(CBL(in_channels, out_channels, 1), CBL(out_channels, in_channels, 3), CBL(in_channels, out_channels, 1),
                              CBL(out_channels, in_channels, 3), CBL(in_channels, out_channels, 1))
        return CBLx5


class tiny(nn.Module):
    def __init__(self, num_class):
        super(tiny, self).__init__()
        self.num_class = num_class
        # 256*512
        self.anchors = [116, 68], [204, 88], [482, 135], [52, 29], [53, 57], [85, 42], [17, 17], [30, 22], [31, 39]
        # 512*1024
        #self.anchors = [222, 136], [408, 176], [965, 270], [104, 59], [106, 115], [170, 84], [34, 33], [59, 44], [63, 78]
        self.layer_detail = [[32, self.anchors[0:3]], [16, self.anchors[3:6]], [8, self.anchors[6:9]]]
        self.DarkNet53 = BackoneNet()
        self.FPN = FPN(num_class, num_anchor=3)

    def forward(self, x):
        tmp1, tmp2, tmp3 = self.DarkNet53(x)
        output = self.FPN(tmp1, tmp2, tmp3)
        for i in range(len(output)):
            x = output[i]
            #stride, anchor = self.layer_stride[i], self.layer_anchor[i]
            stride, anchor = self.layer_detail[i]
            grid = torch.zeros(1)
            bs, _, ny, nx = x.shape
            x = x.view(x.size(0), len(anchor), (4 + 1 + self.num_class), ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if grid.shape[2:4] != x.shape[2:4]:
                    grid = self._make_grid(nx, ny).to(x.device)

                anchor_grid = torch.tensor(anchor).clone().view(1, -1, 1, 1, 2).to(x.device)
                grid = self._make_grid(nx, ny).to(x.device)
                x[..., 0:2] = (x[..., 0:2].sigmoid() + grid) * stride  # xy
                x[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_grid  # wh
                x[..., 4:] = x[..., 4:].sigmoid()
                x = x.view(bs, -1, (4 + 1 + self.num_class))
            output[i] = x
        return output if self.training else torch.cat(output, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
