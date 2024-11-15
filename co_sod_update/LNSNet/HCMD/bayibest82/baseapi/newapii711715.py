import torch
from torch import nn
import torch.nn.functional as F

from 轨道缺陷检测.model.HCMD.Dynamicconvolution.Dynamic.dynamic_conv import Dynamic_conv2d
from 轨道缺陷检测.model.HCMD.bayibest82.baseapi.newapii7122 import attention2d
from 轨道缺陷检测.model.HCMD.bayibest82.baseapi.newapii713 import NSA

class CAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_r = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_f = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb, fuse):
        B, N, C = rgb.shape
        qkv_r = self.qkv_r(rgb).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_f = self.qkv_f(fuse).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qf, kf, vf = qkv_f[0], qkv_f[1], qkv_f[2]  # make torchscript happy (cannot use tensor as tuple)

        attn_r = (qf @ kr.transpose(-2, -1)) * self.scale
        attn_r = attn_r.softmax(dim=-1)
        attn_r = self.attn_drop(attn_r)
        rgb_a = (attn_r @ vr).transpose(1, 2).reshape(B, N, C)
        rgb_a = self.proj(rgb_a) + rgb
        rgb_a = self.proj_drop(rgb_a)

        attn_f = (qr @ kf.transpose(-2, -1)) * self.scale
        attn_f = attn_f.softmax(dim=-1)
        attn_f = self.attn_drop(attn_f)
        fuse_a = (attn_f @ vf).transpose(1, 2).reshape(B, N, C)
        fuse_a = self.proj(fuse_a) + fuse
        fuse_a = self.proj_drop(fuse_a)
        return rgb_a, fuse_a

class AM2(nn.Module):
    def __init__(self, in_channel1, in_channel2, in_channel4):
        super(AM2, self).__init__()
        self.u1 = BasicConv2d(in_channel1, 96, 3, 1, 1)
        self.u2 = BasicConv2d(in_channel2, 96, 3, 1, 1)
        self.u4 = BasicConv2d(in_channel4, 96, 3, 1, 1)
        self.u5 = BasicConv2d(96 * 3, 96, 3, 1, 1)
        self.u6 = BasicConv2d(96, 96, 3, 1, 1)

    def forward(self, x1, x2, x4):
        gm1 = self.u1(x1)
        gm2 = self.u2(x2)
        r2 = self.u4(x4)
        br = F.interpolate(r2, scale_factor=2)
        gm2 = F.interpolate(gm2, scale_factor=4)
        res = torch.cat((gm1, gm2, br), dim=1)
        res = self.u6(self.u5(res) + gm1)
        return res


class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=1):
        """
        更像是DenseNet的Block，从而构造特征内的密集连接
        """
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)


class GM2(nn.Module):
    def __init__(self, in_channel1, in_channel2, size):
        super(GM2, self).__init__()
        self.attention = attention2d(in_planes=in_channel1 * 2, K=in_channel1, temperature=34, ratios=0.25)
        self.basc = BasicConv2d(in_planes=in_channel1 * 2, out_planes=in_channel1, kernel_size=1)
        self.size = size
        self.maxpool2 = nn.AdaptiveMaxPool2d(int(size / 2))
        self.maxpool4 = nn.AdaptiveMaxPool2d(int(size / 4))
        # self.maxpool8 = nn.AdaptiveMaxPool2d(int(size / 8))
        # self.dy1 = Dynamic_conv2d(in_channel1, in_channel1, 1, 1, 1)
        # self.dy2 = Dynamic_conv2d(in_channel1, in_channel1, 1, 1, 1)
        # self.dy3 = Dynamic_conv2d(in_channel1, in_channel1, 1, 1, 1)
        self.dy1 = BasicConv2d(in_channel1, in_channel1, 1)
        self.dy2 = BasicConv2d(in_channel1, in_channel1, 1)
        # self.dy3 = BasicConv2d(in_channel1, in_channel1, 1)

        # self.dens1 = DenseLayer(in_channel1, out_C=in_channel1 * 4)
        self.dens2 = DenseLayer(in_channel1, out_C=in_channel1 * 4)
        self.dens3 = DenseLayer(in_channel1 * 2, out_C=in_channel1 * 4)

        self.maxpool2_a = nn.AdaptiveAvgPool2d(int(size / 2))
        self.maxpool4_a = nn.AdaptiveAvgPool2d(int(size / 4))
        # self.maxpool8_a = nn.AdaptiveAvgPool2d(int(size / 8))
        # self.dy1_a = Dynamic_conv2d(in_channel1, in_channel1, 1, 1, 1)
        # self.dy2_a = Dynamic_conv2d(in_channel1, in_channel1, 1, 1, 1)
        # self.dy3_a = Dynamic_conv2d(in_channel1, in_channel1, 1, 1, 1)
        self.dy1_a = BasicConv2d(in_channel1, in_channel1, 1)
        self.dy2_a = BasicConv2d(in_channel1, in_channel1, 1)
        # self.dy3_a = BasicConv2d(in_channel1, in_channel1, 1)

        self.merge1 = BasicConv2d(in_planes=in_channel1 * 2, out_planes=in_channel1, kernel_size=1)
        self.merge2 = BasicConv2d(in_planes=in_channel1 * 2, out_planes=in_channel1, kernel_size=1)
        # self.merge3 = BasicConv2d(in_planes=in_channel1 * 2, out_planes=in_channel1, kernel_size=1)

    def forward(self, x1, x2, temperature):
        # print(temperature)
        r = torch.cat((x1, x2), dim=1)
        res = self.basc(r)
        res = self.attention(r, temperature).unsqueeze(-1).unsqueeze(-1) * res

        res1 = self.maxpool2(res)
        res2 = self.maxpool4(res)
        # res3 = self.maxpool8(res)
        res1 = self.dy1(res1)
        res2 = self.dy2(res2)
        # res3 = self.dy3(res3)

        res1_a = self.maxpool2_a(res)
        res2_a = self.maxpool4_a(res)
        # res3_a = self.maxpool8_a(res)
        res1_a = self.dy1_a(res1_a)
        res2_a = self.dy2_a(res2_a)
        # res3_a = self.dy3_a(res3_a)

        res1 = self.merge1(torch.cat((res1, res1_a), dim=1))
        res2 = self.merge2(torch.cat((res2, res2_a), dim=1))
        # res3 = self.merge3(torch.cat((res3, res3_a), dim=1))

        # res3 = self.dens1(res3)
        # res3 = F.pixel_shuffle(res3, 2)
        # res23 = torch.cat((res3, res2), dim=1)
        res2 = self.dens2(res2)
        res2 = F.pixel_shuffle(res2, 2)
        res12 = torch.cat((res2, res1), dim=1)
        res12 = self.dens3(res12)
        res12 = F.pixel_shuffle(res12, 2)
        return res12


class node(nn.Module):
    def __init__(self, in_channel1, in_channel2, size1, size2):
        super(node, self).__init__()
        self.ca1new = attention2d(int(in_channel1 + in_channel2 / 4), 0.25, K=int(in_channel1 + in_channel2 / 4),
                                  temperature=34)  # !!!!!
        self.conv1 = BasicConv2d(int(in_channel1 + in_channel2 / 4), in_channel1, 3, 1, 1)

        self.nsa1 = NSA(in_channel2, in_channel1, size=size1)
        self.conv2 = BasicConv2d(in_channel1, in_channel1, 3, 1, 1)
        self.rd1c = BasicConv2d(in_channel1, in_channel2, 3, 2, 1)

        self.yuc = BasicConv2d(in_channel2, in_channel1, 3, 1, 1)
        self.size1 = size1
        self.size2 = size2
        self.d_ls = int(self.size1 / self.size2)

    def forward(self, d1, d2, r1, r2, temperature): # input: r d
        rlayer_featuresx = F.pixel_shuffle(r2, self.d_ls)
        x12 = torch.cat((r1, rlayer_featuresx), dim=1)
        x12c = self.ca1new(x12, temperature).unsqueeze(-1).unsqueeze(-1)
        x12cc = x12c * x12 #   w
        x12c = self.conv1(x12cc)

        y2 = self.nsa1(d2, temperature)
        y12 = y2 * d1 + d1 # w
        y12c = self.conv2(y12)
        rd2 = self.rd1c(y12c)

        rd1 = y12c * x12c
        return rd1, rd2, y12, x12cc


class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, size=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.number = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(self.number)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
