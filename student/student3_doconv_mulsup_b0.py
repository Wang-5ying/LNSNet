from collections import OrderedDict

import torch
from torch import nn

from torchvision.models import mnasnet1_0
import torch.nn.functional as F
from codes.bayibest82segformerbest.best.distill.teacher.newresdecoder4a614t4615622xiuz747117157261015cam11021108110911151116 import \
    Bottleneck2D, SA, CA, BasicConv2d, CAAM_WBY, Channel_aware_CoordAtt, FM, Bottleneck2D_D
from mmseg.models import build_segmentor
from mmcv import Config

from torch.nn import Module, Conv1d, ReLU, Parameter, Softmax
from plug_and_play_modules.DO_Conv.do_conv_pytorch_1_10 import DOConv2d

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


class S(nn.Module):
    def load_pre(self, pre_model1):
        self.resnet.load_state_dict(torch.load(pre_model1)["state_dict"], strict=True)

    def __init__(self, mode='small'):
        super(S, self).__init__()
        # self.student = mnasnet1_0(pretrained=True)
        cfg_path = '/home/wby/PycharmProjects/Transformer_backbone/SegFormerguanfnag/local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py'
        cfg = Config.fromfile(cfg_path)
        self.resnet = build_segmentor(cfg.model)
        self.config = Config()
        self.s1 = nn.Conv2d(128, 1, 1)

        self.sa1x = SA(kernel_size=7)
        self.sa2x = SA(kernel_size=7)
        self.sa3x = SA(kernel_size=7)

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

        self.gm1 = CAAM_WBY(512, 1, [4, 4], norm_layer=nn.BatchNorm2d)

        #（
        # 普通卷积
        self.u1 = nn.Conv2d(512, 320, 3, 1, 1)
        self.u2 = nn.Conv2d(320, 128, 3, 1, 1)
        self.u3 = nn.Conv2d(128, 64, 3, 1, 1)

        # DO_Conv
        self.u1 = DOConv2d(512, 320, kernel_size=3, stride=1, padding=1)
        self.u2 = DOConv2d(320, 128, kernel_size=3, stride=1, padding=1)
        self.u3 = DOConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # ）
        self.ta1 = Channel_aware_CoordAtt(64, 64, 64, 64)
        self.ta2 = Channel_aware_CoordAtt(128, 128, 32, 32)
        self.ta3 = Channel_aware_CoordAtt(320, 320, 16, 16)

        # （
        self.bc1 = nn.Conv2d(512, 320, 3, 1, 1)
        self.bc2 = nn.Conv2d(512, 128, 3, 1, 1)
        self.bc3 = nn.Conv2d(512, 64, 3, 1, 1)

        self.bc1 = DOConv2d(512, 320, 3, 1, 1)
        self.bc2 = DOConv2d(512, 128, 3, 1, 1)
        self.bc3 = DOConv2d(512, 64, 3, 1, 1)
        # ）

        self.fm1 = FM(in_planes=320)
        self.fm2 = FM(in_planes=128)
        self.fm3 = FM(in_planes=64)

        self.bot1 = Bottleneck2D(320, 128)
        self.bot1_1 = Bottleneck2D_D(320, 128)

        self.bot2 = Bottleneck2D_D(128, 64)
        self.bot2_1 = Bottleneck2D_D(128, 64)

        self.bot3 = Bottleneck2D_D(64, 32)
        self.bot3_1 = Bottleneck2D_D(64, 32)

        self.bot4 = Bottleneck2D_D(64, 32)
        self.bot4_1 = Bottleneck2D_D(64, 32)

        self.u4 = nn.Conv2d(64, 1, 1)

        # self.o1 = BasicConv2d(512, 320, 1)
        # self.o2 = BasicConv2d(320, 128, 1)
        # self.o3 = BasicConv2d(128, 64, 1)
        self.sup2 = BasicConv2d(128, 1, 3, 1, 1)
        self.sup3 = BasicConv2d(64, 1, 3, 1, 1)
    def forward(self, r, d, t):
        # print(r.size(),d.size())
        # r
        global cam_cls
        features_r = []
        features_d = []
        features_r = self.resnet.extract_feat(r)
        features_d = self.resnet.extract_feat(d)
        # for i in features_d:
        #     print(i.size())

        # pre-decoder2   ---> AFF
        # 1
        g1_d = self.ca1(self.beforeca1(features_r[0]))
        g2_d = self.beforeca2(self.ca2(g1_d + features_r[1], features_r[1]))
        g3_d = self.beforeca3(self.ca3(g2_d + features_r[2], features_r[2]))
        g4_d = self.beforeca4(self.ca4(g3_d + features_r[3], features_r[3]))

        g1_r = self.ca1(self.beforeca1(features_d[0]))
        g2_r = self.beforeca2(self.ca2(g1_r + features_d[1], features_d[1]))
        g3_r = self.beforeca3(self.ca3(g2_r + features_d[2], features_d[2]))
        g4_r = self.beforeca4(self.ca4(g3_r + features_d[3], features_d[3]))

        # 2
        # g1_d = self.ca1(self.beforeca1(features_r[0]))
        # g2_d = self.beforeca2(self.ca2(g1_d, features_r[1]))
        # g3_d = self.beforeca3(self.ca3(g2_d, features_r[2]))
        # g4_d = self.beforeca4(self.ca4(g3_d, features_r[3]))
        #
        # g1_r = self.ca1(self.beforeca1(features_d[0]))
        # g2_r = self.beforeca2(self.ca2(g1_r, features_d[1]))
        # g3_r = self.beforeca3(self.ca3(g2_r, features_d[2]))
        # g4_r = self.beforeca4(self.ca4(g3_r, features_d[3]))

        bc = self.gm1(g4_r, g4_d)  ## ([8, 512, 8, 8])

        if self.training:
            # cls = self.classifier(self.avgpool(bc1).squeeze(-1).squeeze(-1))
            cam_cls = bc[1]
        # pre-decoder1
        # 1
        features_d[-2] = F.interpolate(self.sa1x(self.ca4_4(features_d[-1], features_d[-1])), scale_factor=2) * \
                         features_d[-2] + features_d[-2]
        features_d[-3] = F.interpolate(self.sa2x(self.ca3_3(features_d[-2], features_d[-2])), scale_factor=2) * \
                         features_d[-3] + features_d[-3]
        features_d[-4] = F.interpolate(self.sa1x(self.ca2_2(features_d[-3], features_d[-3])), scale_factor=2) * \
                         features_d[-4] + features_d[-4]

        features_r[-2] = F.interpolate(self.sa1x(self.ca4_4(features_r[-1], features_r[-1])), scale_factor=2) * \
                         features_r[-2] + features_r[-2]
        features_r[-3] = F.interpolate(self.sa2x(self.ca3_3(features_r[-2], features_r[-2])), scale_factor=2) * \
                         features_r[-3] + features_r[-3]
        features_r[-4] = F.interpolate(self.sa1x(self.ca2_2(features_r[-3], features_r[-3])), scale_factor=2) * \
                         features_r[-4] + features_r[-4]
        # # 2
        # features_d[-2] = self.o1(F.interpolate(self.ca4_4(features_d[-1], features_d[-1]), scale_factor=2)) * \
        #                  features_d[-2] + features_d[-2]
        # features_d[-3] = self.o2(F.interpolate(self.ca3_3(features_d[-2], features_d[-2]), scale_factor=2)) * \
        #                  features_d[-3] + features_d[-3]
        # features_d[-4] = self.o3(F.interpolate(self.ca2_2(features_d[-3], features_d[-3]), scale_factor=2)) * \
        #                  features_d[-4] + features_d[-4]
        #
        # features_r[-2] = self.o1(F.interpolate(self.ca4_4(features_r[-1], features_r[-1]), scale_factor=2)) * \
        #                  features_r[-2] + features_r[-2]
        # features_r[-3] = self.o2(F.interpolate(self.ca3_3(features_r[-2], features_r[-2]), scale_factor=2)) * \
        #                  features_r[-3] + features_r[-3]
        # features_r[-4] = self.o3(F.interpolate(self.ca2_2(features_r[-3], features_r[-3]), scale_factor=2)) * \
        #                  features_r[-4] + features_r[-4]

        # decoder
        x1 = self.u1(bc[0])
        res1 = F.interpolate(x1, 16)
        res1 = res1 + self.ta3(features_r[-2]) * features_r[-2] + self.ta3(features_d[-2]) * features_d[-2]
        bc1 = F.interpolate(self.bc1(bc[0]), size=16)
        res1 = res1 * bc1 + res1
        res1_1 = self.fm1(self.fm1(res1))
        res1_2 = self.bot1(res1)
        res1 = self.bot1_1(res1_1 + res1_2)
        # res1, res1_score = self.cam1(res1)
        # res1 = self.cam1_c(res1)

        x2 = self.u2(res1)
        res2 = F.interpolate(x2, 32)
        res2 = res2 + self.ta2(features_r[-3]) * features_r[-3] + self.ta2(features_d[-3]) * features_d[-3]
        bc2 = F.interpolate(self.bc2(bc[0]), size=32)
        res2 = res2 * bc2 + res2
        res2_1 = self.fm2(self.fm2(res2))
        res2_2 = self.bot2(res2)
        res2 = self.bot2_1(res2_1 + res2_2)
        # res2, res1_score = self.cam2(res2)
        # res2 = self.cam2_c(res2)

        x3 = self.u3(res2)
        res3 = F.interpolate(x3, 64)
        res3 = res3 + self.ta1(features_r[-4]) * features_r[-4] + self.ta1(features_d[-4]) * features_d[-4]
        bc3 = F.interpolate(self.bc3(bc[0]), size=64)
        res3 = res3 * bc3 + res3
        res3_1 = self.fm3(self.fm3(res3))
        res3_2 = self.bot3(res3)
        res3 = self.bot3_1(res3_1 + res3_2)
        # res3, res1_score = self.cam3(res3)
        # res3 = self.cam3_c(res3)

        res4 = self.bot4_1(self.bot4(res3))
        res4 = self.u4(res4)
        res4 = F.interpolate(res4, 256)

        soft = res4 / 4
        res2 = F.interpolate(self.sup2(res2), 256)
        res3 = F.interpolate(self.sup3(res3), 256)
        soft2 = res2 / 4
        soft3 = res3 / 4
        if self.training:
            soft4 = cam_cls / 4

        # print(res2.size(), res3.size(),res4.size(),soft2.size(),soft3.size(),soft.size(),cam_cls.size(),soft4.size())
        if self.training:
            # return res2, res3, res4, soft2, soft3, soft, cam_cls, soft4
            return res2, soft2, res3, soft3, res4, soft, cam_cls, soft4
        else:
            return res4, res4


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    model = S()
    # model.load_pre("/media/wby/KINGSTON/mnasnet-a1-140/model.ckpt")
    model.cuda()
    out = model(a, b, 1)
    # for i in out:
    #     print(i.shape)
