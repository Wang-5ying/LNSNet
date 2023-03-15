import importlib
# 1k(0.0648) 150(0.0497) 183(0.0661)
from math import sqrt

import numpy as np
import torch
from mmseg.models import build_segmentor
from torch import nn
import torch.nn.functional as F
# from MLNet.ResT.models.rest_v2 import ResTV2
from codes.bayibest82.baseapi.newapii711715 import BasicConv2d, node, CA, GM2  # atttention
# from codes.bayibest82segformerbest.External_Attention.model.attention.ExternalAttention import ExternalAttention
# from bayibestsegformer.baseapi.newapii711715 import BasicConv2d, node, CA, GM2  # atttention
# from MLNet.mobilevit import mobilevit_s
from mmcv import Config
from torchvision.models import vgg16, vgg16_bn
from collections import OrderedDict
from plug_and_play_modules.DO_Conv.do_conv_pytorch_1_10 import DOConv2d
from timm.models.layers import DropPath, trunc_normal_
# from mmseg.models.backbones.mix_transformer import mit_b2
from mmseg.models.backbones.mix_transformer import mit_b1
# from mmseg.models import build_segmentor
#############630 qudiaol dongtaijuanji   vt5000 0.026 150epoch
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        # print("beforerpooling,",x.size())
        y = self.pool(x) - x
        # print("Afterpool", y.size())
        return y


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FM(nn.Module):
    def __init__(self, in_planes, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.):
        super(FM, self).__init__()
        self.norm1 = norm_layer(in_planes)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(in_planes)
        mlp_hidden_dim = int(in_planes * mlp_ratio)
        self.mlp = Mlp(in_features=in_planes, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

    def forward(self, x):
        # print("x2",x.size())
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        # print("x2",x.size())
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print("x3",x.size())
        return x


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


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x, temperature):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / temperature, 1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Channel_aware_CoordAtt(nn.Module):
    def __init__(self, inp, oup, h, w, reduction=32):
        super(Channel_aware_CoordAtt, self).__init__()
        self.h = h
        self.w = w
        self.pool_h = nn.AdaptiveAvgPool2d((h, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, w))
        self.pool_c = nn.AdaptiveAvgPool2d((w, 1))

        mip = max(8, (inp + self.h) // reduction)

        self.conv1 = nn.Conv2d(inp + self.h, mip, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(inp + self.h, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_y1 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_y2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        channel = x.reshape(n, h, w, c)
        x_c = self.pool_c(channel)

        temp = x_c.permute(0, 2, 1, 3)
        y1 = torch.cat([x_h, temp], dim=1)
        y1 = self.conv1(y1)
        y1 = self.bn1(y1)
        y1 = self.act(y1)

        y2 = torch.cat([x_w, x_c], dim=1)
        y2 = self.conv2(y2)
        y2 = self.bn1(y2)
        y2 = self.act(y2).permute(0, 1, 3, 2)

        y1 = self.conv_y1(y1).sigmoid()

        y2 = self.conv_y2(y2).sigmoid()
        # y2_w = self.conv_y2w(y2_w).sigmoid()

        # 如果下面这个原论文代码用不了的话，可以换成另一个试试
        out = identity * y1 * y2
        # out = a_h.expand_as(x) * a_w.expand_as(x) * identity

        return out


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h


class Gru(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):
        super(Gru, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = BasicConv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        self.conv_state2 = BasicConv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj2 = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)
        # fusion
        self.fc_2 = nn.Conv2d(num_in, num_in, kernel_size=kernel_size, padding=padding, stride=(1, 1),
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, x, y):
        batch_size = x.size(0)
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
        y_proj_reshaped = self.conv_proj(y).view(batch_size, self.num_n, -1)
        x_state_2 = self.conv_state2(x).view(batch_size, self.num_s, -1)
        x_n_state1 = torch.bmm(x_state_reshaped, y_proj_reshaped.permute(0, 2, 1))
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        x_n_rel1 = self.gcn1(x_n_state2)
        x_n_rel2 = self.gcn2(x_n_rel1)
        x_state_reshaped = torch.bmm(x_n_rel2.permute(0, 2, 1), x_state_2)
        x_state = x_state_reshaped.view(batch_size, 96, 8, 8)
        out = x + self.blocker(self.fc_2(x_state)) + y
        return out


def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B, -1, rH, rW, C)  # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out


def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W)  # [B, C, H, W]
    return out


class GCN_CAM(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN_CAM, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class MAttention(nn.Module):
    def __init__(self, dim, reduction=8, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv_r = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_f = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, rgb):
        B, C, H, W = rgb.shape
        rgb = rgb.reshape(B, H * W, C)
        B, N, C = rgb.shape

        qkv_r = self.qkv_r(rgb).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]  # make torchscript happy (cannot use tensor as tuple)
        qkv_g = self.qkv_f(rgb).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qg, kg, vg = qkv_g[0], qkv_g[1], qkv_g[2]  # make torchscript happy (cannot use tensor as tuple)
        attn_r = (qr @ kg.transpose(-2, -1)) * self.scale
        # attn_r = (qf @ kr) * self.scale
        attn_r = attn_r.softmax(dim=-1)
        attn_r = self.attn_drop(attn_r)
        rgb_a = (attn_r @ vg).transpose(1, 2).reshape(B, N, C)
        rgb_a = self.proj(rgb_a)
        rgb_a = self.proj_drop(rgb_a)

        B, N, C = rgb_a.shape
        rgb_a = rgb_a.reshape(B, C, int(sqrt(N)), int(sqrt(N)))
        # print(rgb_a.size())
        return rgb_a


class CAAM(nn.Module):
    """
    Class Activation Attention Module
    """

    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN_CAM(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        # bin_num_h =S
        cam = self.conv_cam(self.dropout(x))  # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam))  # [B, K, bin_num_h, bin_num_w]

        residual = x  # [B, C, H, W]
        cam = patch_split(cam, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, K]
        # print("cam", cam.size())
        x = patch_split(x, self.bin_size)  # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH * rW, K)  # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH * rW, C)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B, K, -1).transpose(1, 2).unsqueeze(3)  # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3),
                                   x) * bin_confidence  # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats)  # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats)  # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1)  # [B, bin_num_h * bin_num_w, K, C]

        query = self.proj_query(x)  # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats)  # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats)  # [B, bin_num_h * bin_num_w, K, C//2]

        aff_map = torch.matmul(query, key.transpose(2, 3))  # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value)  # [B, bin_num_h * bin_num_w, rH * rW, C]

        out = out.view(B, -1, rH, rW, value.shape[-1])  # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size)  # [B, C, H, W]

        out = residual + self.conv_out(out)
        return out, cls_score


class CAAM_WBY(nn.Module):
    """
    Class Activation Attention Module
    """

    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM_WBY, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        # self.dropout = nn.Dropout2d(0.1)
        # 1
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()
        # 2
        self.conv_cam_y = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam_y = nn.AdaptiveAvgPool2d(bin_size)

        bin_num = bin_size[0] * bin_size[1]
        # 1
        self.gcn = GCN_CAM(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)

        # 2
        self.gcn_y = GCN_CAM(bin_num, feat_in)
        self.fuse_y = nn.Conv2d(bin_num, 1, kernel_size=1)

        # 1
        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_in, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.ReLU(inplace=True)
        self.msa = MAttention(feat_in)
        # 2
        self.conv_out_y = nn.Sequential(
            nn.Conv2d(feat_in, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.ReLU(inplace=True)
        )
        self.relu_y = nn.ReLU(inplace=True)
        self.msa_y = MAttention(feat_in)
        # print("feat_in", feat_in)

    def forward(self, x, y):
        ### 1
        residule = x
        cam1 = self.conv_cam(x)  # [B, K, H, W]
        cam = cam1
        cls_score = self.sigmoid(self.pool_cam(cam))  # [B, K, bin_num_h, bin_num_w]
        ms = self.msa(residule)
        out = self.conv_out(ms)  # 1017 去掉残差  + residule   🐶

        ### 2
        residule_y = y
        cam1_y = self.conv_cam_y(y)  # [B, K, H, W]
        cam_y = cam1_y
        cls_score_y = self.sigmoid(self.pool_cam_y(cam_y))  # [B, K, bin_num_h, bin_num_w]

        ms_y = self.msa_y(residule_y)
        out_y = self.conv_out_y(ms_y)  # 1017 去掉残差  + residule   🐶
        out = out + out_y
        cls_score = cls_score + cls_score_y
        return out, cls_score


class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, inplanes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck2D_D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D_D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = DOConv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = DOConv2d(planes, planes, kernel_size=3,
                              stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = DOConv2d(planes, inplanes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck2DC(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2DC, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual

        return out


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
        return self.sigmoid(x) * x

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class Dp(nn.Module):
    def __init__(self, inchannel1, inchannel, ourchannel, size):
        super(Dp, self).__init__()
        self.bc = int(inchannel/2)
        self.pre1 = Bottleneck2D(inchannel, self.bc)
        self.pre2 = Bottleneck2D(inchannel, self.bc)
        self.u1 = nn.Conv2d(inchannel1, inchannel, 3, 1, 1)
        self.trans4 = torch.nn.ConvTranspose2d(inchannel, inchannel, 3, 2, 1, 1, 1, True, 1)
        self.sig = nn.Sigmoid()
        self.trans6 = torch.nn.ConvTranspose2d(inchannel1, inchannel, 3, 2, 1, 1, 1, True, 1)
        self.bc1 = nn.Conv2d(inchannel1, inchannel, 3, 1, 1)
        self.trans5 = torch.nn.ConvTranspose2d(inchannel, inchannel, 3, 2, 1, 1, 1, True, 1)
        self.fm1 = FM(in_planes=inchannel)
        self.bot1 = Bottleneck2D(inchannel, self.bc)
        self.bot1_1 = Bottleneck2D(inchannel, self.bc)
        self.ta3 = Channel_aware_CoordAtt(inchannel, inchannel, size, size)
        self.out = BasicConv2d(inchannel, ourchannel, 3, 1, 1)
    def forward(self, res, bc, r, d):
        x1 = self.u1(bc)
        x1 = self.sig(self.trans4(x1))
        res = self.trans6(res)
        # print(x1.size(), res.size())
        res1 = x1 * res + res
        res1 = res1 + self.ta3(r) * r + self.ta3(d) * d
        bc1 = self.sig(self.trans5(self.bc1(bc)))
        res1 = res1 * bc1 + res1
        res1_1 = self.fm1(self.fm1(res1))
        res1_2 = self.bot1(res1)
        res = self.out(self.bot1_1(res1_1 + res1_2))
        return x1, res
class M(nn.Module):
    def load_pre(self, pre_model1):

        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.decoder.load_state_dict(new_state_dict3, strict=False)
        self.resnet.load_state_dict(new_state_dict3, strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model1}")
        print(f"Depth SwinTransformer loading pre_model ${pre_model1}")

    def __init__(self, mode='small'):
        super(M, self).__init__()
        # cfg_path = '/home/wby/PycharmProjects/Transformer_backbone/SegFormerguanfnag/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py'
        # cfg = Config.fromfile(cfg_path)
        self.resnet = mit_b1()
        self.config = Config()
        self.decoder = mit_b1()

        self.u4 = nn.Conv2d(64, 3, 1)
        # self.gm1 = GM2(512, 512, 8)

        self.sup1 = BasicConv2d(320, 1, 3, 1, 1)
        self.sup2 = BasicConv2d(128, 1, 3, 1, 1)
        self.sup3 = BasicConv2d(64, 1, 3, 1, 1)

        # change to B
        self.node1 = node(64, 128, 64, 32)
        self.node2 = node(320, 512, 16, 8)

        self.ta1 = Channel_aware_CoordAtt(64, 64, 64, 64)
        self.ta2 = Channel_aware_CoordAtt(128, 128, 32, 32)
        self.ta3 = Channel_aware_CoordAtt(320, 320, 16, 16)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 291)

        self.bc1 = nn.Conv2d(512, 320, 3, 1, 1)
        self.bc2 = nn.Conv2d(512, 128, 3, 1, 1)
        self.bc3 = nn.Conv2d(512, 64, 3, 1, 1)

        # self.gm1 = CAAM(512, 1, [4, 4], norm_layer=nn.BatchNorm2d)
        self.gm1_c = BasicConv2d(512, 512, 4, 2, 1, 1)
        self.gm1 = CAAM_WBY(512, 1, [32, 32], norm_layer=nn.BatchNorm2d)
        self.gm1_2 = CAAM_WBY(320, 1, [32, 32], norm_layer=nn.BatchNorm2d)
        self.gm1_3 = CAAM_WBY(128, 1, [64, 64], norm_layer=nn.BatchNorm2d)
        self.cam2 = BasicConv2d(320, 320, 1)
        self.cam3 = BasicConv2d(128, 128, 1)


        self.beforeca1 = BasicConv2d(64, 128, 3, 2, 1)  # 往pre_decoder里加do_conv效果会变差
        self.ca1 = CA(128)
        self.ca1_1 = AFF(128)
        # self.ca1 = AFF(128)
        self.beforeca2 = BasicConv2d(128, 320, 3, 2, 1)
        # self.ca2 = CA(320)
        self.ca2 = AFF(128)
        self.ca2_2 = AFF(128)
        self.beforeca3 = BasicConv2d(320, 512, 3, 2, 1)
        # self.ca3 = CA(512)
        self.ca3 = AFF(320)
        self.ca3_3 = AFF(320)
        self.beforeca4 = BasicConv2d(512, 512, 3, 1, 1)
        # self.ca4 = CA(512)
        self.ca4 = AFF(512)
        self.ca4_4 = AFF(512)

        self.md1 = Bottleneck2D(512, 512)
        self.md2 = Bottleneck2D(320, 320)
        self.md3 = Bottleneck2D(128, 128)
        self.md4 = Bottleneck2D(64, 64)

        self.mr1 = Bottleneck2D(512, 512)
        self.mr2 = Bottleneck2D(320, 320)
        self.mr3 = Bottleneck2D(128, 128)
        self.mr4 = Bottleneck2D(64, 64)

        self.sa1x = SA(kernel_size=7)
        # self.sa1xc = BasicConv2d(512,320,1)

        self.sa2x = SA(kernel_size=7)
        # self.sa2xc = BasicConv2d(320, 128, 1)

        self.sa3x = SA(kernel_size=7)
        # self.sa3xc = BasicConv2d(128, 64, 1)
        self.dp1 = Dp(512, 320, 320, 16)
        self.dp2 = Dp(320, 128, 128, 32)
        self.dp3 = Dp(128, 64, 64, 64)

        self.trans1 = torch.nn.ConvTranspose2d(1, 1, 3, 2, 1, 1, 1, True, 1)
        self.trans2 = torch.nn.ConvTranspose2d(1, 1, 3, 2, 1, 1, 1, True, 1)
        self.trans3 = torch.nn.ConvTranspose2d(1, 1, 3, 2, 1, 1, 1, True, 1)

        self.bot4_1 = Bottleneck2D(64, 32)
        self.bot4 = Bottleneck2D(64, 32)

        self.trans8 = torch.nn.ConvTranspose2d(64, 1, 3, 2, 1, 1, 1, True, 1)

        # decoder
        self.deco1 = torch.nn.ConvTranspose2d(512, 320, 3, 2, 1, 1, 1, True, 1)
        self.deco2 = torch.nn.ConvTranspose2d(320, 128, 3, 2, 1, 1, 1, True, 1)
        self.deco3 = torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1, 1, True, 1)

        # guide
        self.gsa1 = SA(7)
        self.gsa2 = SA(7)
        self.gsa3 = SA(7)

        self.deco4 = torch.nn.ConvTranspose2d(3, 3, 3, 2, 1, 1, 1, True, 1)
        self.deco5 = torch.nn.ConvTranspose2d(3, 3, 3, 2, 1, 1, 1, True, 1)
        self.si = nn.Sigmoid()

        # return flow
        self.before1 = BasicConv2d(512, 320, 3, 1, 1)
        self.r1 = BasicConv2d(320, 128, 3, 1, 1)
        self.before2 = BasicConv2d(320, 128, 3, 1, 1)
        self.r2 = BasicConv2d(128, 64, 3, 1, 1)
        self.before3 = BasicConv2d(128, 64, 3, 1, 1)
        self.r3 = BasicConv2d(64, 3, 3, 1, 1)

        self.before1t = BasicConv2d(512, 320, 3, 1, 1)
        self.t1 = BasicConv2d(320, 128, 3, 1, 1)
        self.before2t = BasicConv2d(320, 128, 3, 1, 1)
        self.t2 = BasicConv2d(128, 64, 3, 1, 1)
        self.before3t = BasicConv2d(128, 64, 3, 1, 1)
        self.t3 = BasicConv2d(64, 3, 3, 1, 1)
    def forward(self, r, d):
        global cls, cam_cls
        # rte = self.resnet.forward(r)
        # dte = self.resnet.forward(d)
        B = r.shape[0]
        srte = []
        sdte = []

        global cls, cam_cls
        brte = self.resnet.forward(r)
        bdte = self.resnet.forward(d)
        srte = brte
        sdte = bdte
        rte = []
        dte = []
        # torch.Size([2, 64, 64, 64])
        # torch.Size([2, 128, 32, 32])
        # torch.Size([2, 320, 16, 16])
        # torch.Size([2, 512, 8, 8])
        # <----------------RGB
        r4 = self.before1(F.interpolate(brte[3], 16, mode='bilinear', align_corners=True))
        r3 = self.r1(r4 + brte[2])
        r3 = F.interpolate(r3, 32, mode='bilinear', align_corners=True)
        # stage 3
        x, H, W = self.resnet.patch_embed3(r3)
        for i, blk in enumerate(self.resnet.block3):
            x = blk(x, H, W)
        x = self.resnet.norm3(x)
        r3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        r2 = self.before2(F.interpolate(r3, 32, mode='bilinear', align_corners=True))
        r2 = self.r2(r2 + brte[1])
        r2 = F.interpolate(r2, 64, mode='bilinear', align_corners=True)
        # stage 2
        x, H, W = self.resnet.patch_embed2(r2)
        for i, blk in enumerate(self.resnet.block2):
            x = blk(x, H, W)
        x = self.resnet.norm2(x)
        r2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        r1 = self.before3(F.interpolate(r2, 64, mode='bilinear', align_corners=True))
        r1 = self.r3(r1 + brte[0])
        r1 = F.interpolate(r1, 256, mode='bilinear', align_corners=True)
        # stage 1
        x, H, W = self.resnet.patch_embed1(r1)
        for i, blk in enumerate(self.resnet.block1):
            x = blk(x, H, W)
        x = self.resnet.norm1(x)
        r1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        rte.append(r1)
        rte.append(r2)
        rte.append(r3)
        rte.append(brte[3])



        # <----------------Depth
        d4 = self.before1t(F.interpolate(bdte[3], 16, mode='bilinear', align_corners=True))
        d3 = self.t1(d4 + bdte[2])
        d3 = F.interpolate(d3, 32, mode='bilinear', align_corners=True)
        # stage 3
        x, H, W = self.resnet.patch_embed3(d3)
        for i, blk in enumerate(self.resnet.block3):
            x = blk(x, H, W)
        x = self.resnet.norm3(x)
        d3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        d2 = self.before2t(F.interpolate(d3, 32, mode='bilinear', align_corners=True))
        d2 = self.t2(d2 + bdte[1])
        d2 = F.interpolate(d2, 64, mode='bilinear', align_corners=True)
        # stage 2
        x, H, W = self.resnet.patch_embed2(d2)
        for i, blk in enumerate(self.resnet.block2):
            x = blk(x, H, W)
        x = self.resnet.norm2(x)
        d2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        d1 = self.before3t(F.interpolate(d2, 64, mode='bilinear', align_corners=True))
        d1 = self.t3(d1 + bdte[0])
        d1 = F.interpolate(d1, 256, mode='bilinear', align_corners=True)
        # stage 1
        x, H, W = self.resnet.patch_embed1(d1)
        for i, blk in enumerate(self.resnet.block1):
            x = blk(x, H, W)
        x = self.resnet.norm1(x)
        d1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        dte.append(d1)
        dte.append(d2)
        dte.append(d3)
        dte.append(bdte[3])


        # for i in rte:
        #     print(i.size())
        # torch.Size([4, 64, 64, 64])
        # torch.Size([4, 128, 32, 32])
        # torch.Size([4, 320, 16, 16])
        # torch.Size([4, 512, 8, 8])

        # g1_d = self.ca1(self.beforeca1(dte[0]))
        # g2_d = self.ca2(self.beforeca2(g1_d + dte[1]), dte[1])
        # g3_d = self.ca3(self.beforeca3(g2_d + dte[2]), dte[2])
        # g4_d = self.ca4(self.beforeca4(g3_d + dte[3]), dte[3])
        #
        # g1_r = self.ca1(self.beforeca1(rte[0]))
        # g2_r = self.ca2(self.beforeca2(g1_r + rte[1]), rte[1])
        # g3_r = self.ca3(self.beforeca3(g2_r + rte[2]), rte[2])
        # g4_r = self.ca4(self.beforeca4(g3_r + rte[3]), rte[3])

        g1_d = self.ca1(self.beforeca1(dte[0]))
        g2_d = self.beforeca2(self.ca2(g1_d + dte[1], dte[1]))
        g3_d = self.beforeca3(self.ca3(g2_d + dte[2], dte[2]))
        g4_d = self.beforeca4(self.ca4(g3_d + dte[3], dte[3]))

        g1_r = self.ca1(self.beforeca1(rte[0]))
        g2_r = self.beforeca2(self.ca2(g1_r + rte[1], rte[1]))
        g3_r = self.beforeca3(self.ca3(g2_r + rte[2], rte[2]))
        g4_r = self.beforeca4(self.ca4(g3_r + rte[3], rte[3]))

        bc = self.gm1(g4_r, g4_d)  ## ([8, 512, 8, 8])
        bc_2 = self.gm1_2(rte[2], dte[2])
        bc_3 = self.gm1_3(rte[1], dte[1])
        # print(bc[0].size(), bc_2[0].size(), bc_3[0].size())

        rte[-1] = self.mr1(rte[-1])
        rte[-2] = self.mr2(rte[-2])
        rte[-3] = self.mr3(rte[-3])
        rte[-4] = self.mr4(rte[-4])

        dte[-1] = self.md1(dte[-1])
        dte[-2] = self.md2(dte[-2])
        dte[-3] = self.md3(dte[-3])
        dte[-4] = self.md4(dte[-4])

        # print(rte[-4].size(),rte[-3].size())
        rte[-2] = self.trans1(self.sa1x(rte[-1])) * rte[-2] + rte[-2]
        rte[-3] = self.trans2(self.sa2x(rte[-2])) * rte[-3] + rte[-3]
        rte[-4] = self.trans3(self.sa1x(rte[-3])) * rte[-4] + rte[-4]
        # if self.training:
            # cls = self.classifier(self.avgpool(bc1).squeeze(-1).squeeze(-1))
        cam_cls = bc[1]
        cam_cls_2 = bc_2[1]
        cam_cls_3 = bc_3[1]

        decode = []
        # print("bc",bc.size(),cls.size())
        bc, res1 = self.dp1(rte[-1]+dte[-1],  bc[0], rte[-2], dte[-2])
        guide = self.gsa1(bc)
        # res1 = torch.cat((res1, guide), dim= 1)
        res1 = res1 * guide
        B = res1.shape[0]
        x, H, W = self.decoder.patch_embed4(res1)
        for i, blk in enumerate(self.decoder.block4):
            x = blk(x, H, W)
        x = self.decoder.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res1 = res1 + self.deco1(x)
        decode.append(res1)

        bc, res2 = self.dp2(res1, bc, rte[-3], dte[-3])
        guide = self.gsa2(bc)
        # res2 = torch.cat((res2, guide), dim=1)
        res2 = res2 * guide
        x, H, W = self.decoder.patch_embed3(res2)
        for i, blk in enumerate(self.decoder.block3):
            x = blk(x, H, W)
        x = self.decoder.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res2 = res2 + self.deco2(x)
        decode.append(res2)

        bc, res3 = self.dp3(res2, bc, rte[-4], dte[-4])
        guide = self.gsa3(bc)
        # res3 = torch.cat((res3, guide), dim=1)
        res2 = res2 * F.interpolate(guide, size=32)
        x, H, W = self.decoder.patch_embed2(res3)
        for i, blk in enumerate(self.decoder.block2):
            x = blk(x, H, W)
        x = self.decoder.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res3 = res3 + self.deco3(x)
        decode.append(res3)
        # print(res1.size(), res2.size(), res3.size())



        res4 = self.bot4_1(self.bot4(res3))
        res4 = self.si(self.deco5(self.deco4(self.u4(res4))))
        # print("res4", res4.size())
        # stage 1
        x, H, W = self.decoder.patch_embed1(res4)
        for i, blk in enumerate(self.decoder.block1):
            x = blk(x, H, W)
        x = self.decoder.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        decode.append(x)
        # print("x",x.size(),res4.size())
        res4 = F.interpolate(self.trans8(x), 256)

        res1 = F.interpolate(self.sup1(res1), 256)
        res2 = F.interpolate(self.sup2(res2), 256)
        res3 = F.interpolate(self.sup3(res3), 256)

        # studnet
        # if self.training:
        #     return res1, res2, res3, res4, cam_cls, cam_cls_2, cam_cls_3  ####726
        # else:
        #     # return res2, res3, res4
        #     return res2, res4, res4
        # distill
        soft = res4 / 4
        soft2 = res2 / 4
        soft3 = res3 / 4

        # if self.training:
        soft4 = cam_cls / 4
        returnv =[]
        returnv.append(res2)
        returnv.append(soft2)
        returnv.append(res3)
        returnv.append(soft3)
        returnv.append(res4)
        returnv.append(soft)
        returnv.append(cam_cls)
        returnv.append(soft4)
        if self.training:
            return returnv, srte, sdte, decode
        else:
            # return res2, res3, res4
            return res2, res4, res4


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256)
    b = torch.randn(2, 3, 256, 256)
    model = M()
    # model.cuda()
    out = model(a, b)
