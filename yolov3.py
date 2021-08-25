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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        assert in_channels % 2 == 0
        self.ResUnit = nn.Sequential(CBL(in_channels, in_channels // 2, kernel_size=1), CBL(in_channels // 2, in_channels))

    def forward(self, x):
        residual = x
        out = self.ResUnit(x)
        out += residual
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


class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()

        self.conv1 = CBL(3, 32)
        self.Res1 = self._make_layer_(32, 64, num_blocks=1)
        self.Res2 = self._make_layer_(64, 128, num_blocks=2)
        self.Res3 = self._make_layer_(128, 256, num_blocks=8)
        self.Res4 = self._make_layer_(256, 512, num_blocks=8)
        self.Res5 = self._make_layer_(512, 1024, num_blocks=4)

    def _make_layer_(self, in_channels, out_channels, num_blocks):
        layers = [CBL(in_channels, out_channels, stride=2)]
        for i in range(0, num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.Res1(tmp)
        tmp = self.Res2(tmp)
        out3 = self.Res3(tmp)
        out2 = self.Res4(out3)
        out1 = self.Res5(out2)
        return out1, out2, out3


class FPN(nn.Module):
    def __init__(self, num_class, num_anchor=3):
        super(FPN, self).__init__()
        self.num_class = num_class
        self.CBLx5_1 = self._make_CBLx5_(1024, 512)
        self.predict_1 = nn.Sequential(CBL(512, 1024, 3), nn.Conv2d(1024, (4 + 1 + self.num_class) * num_anchor, 1))
        self.CBL_Up2D_1 = nn.Sequential(CBL(512, 256, 1), Interpolate(scale_factor=2, mode="nearest"))

        self.CBLx5_2 = self._make_CBLx5_(768, 256)
        self.predict_2 = nn.Sequential(CBL(256, 768, 3), nn.Conv2d(768, (4 + 1 + self.num_class) * num_anchor, 1))
        self.CBL_Up2D_2 = nn.Sequential(CBL(256, 128, 1), Interpolate(scale_factor=2, mode="nearest"))

        self.CBLx5_3 = self._make_CBLx5_(384, 128)
        self.predict_3 = nn.Sequential(CBL(128, 384, 3), nn.Conv2d(384, (4 + 1 + self.num_class) * num_anchor, 1))

    def forward(self, tmp1, tmp2, tmp3):
        output = []
        tmp = self.CBLx5_1(tmp1)
        out1 = self.predict_1(tmp)
        tmp = torch.cat((self.CBL_Up2D_1(tmp), tmp2), 1)

        tmp = self.CBLx5_2(tmp)
        out2 = self.predict_2(tmp)
        tmp = torch.cat((self.CBL_Up2D_2(tmp), tmp3), 1)

        tmp = self.CBLx5_3(tmp)
        out3 = self.predict_3(tmp)
        output = [out1, out2, out3]
        return output

    def _make_CBLx5_(self, in_channels, out_channels):
        assert in_channels % 2 == 0
        CBLx5 = nn.Sequential(CBL(in_channels, out_channels, 1), CBL(out_channels, in_channels, 3), CBL(in_channels, out_channels, 1),
                              CBL(out_channels, in_channels, 3), CBL(in_channels, out_channels, 1))
        return CBLx5


class YoloV3(nn.Module):
    def __init__(self, num_class):
        super(YoloV3, self).__init__()
        self.num_class = num_class
        # 256*512
        self.anchors = [116, 68], [204, 88], [482, 135], [52, 29], [53, 57], [85, 42], [17, 17], [30, 22], [31, 39]
        # 512*1024
        #self.anchors = [222, 136], [408, 176], [965, 270], [104, 59], [106, 115], [170, 84], [34, 33], [59, 44], [63, 78]
        self.layer_detail = [[32, self.anchors[0:3]], [16, self.anchors[3:6]], [8, self.anchors[6:9]]]
        self.DarkNet53 = DarkNet53()
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