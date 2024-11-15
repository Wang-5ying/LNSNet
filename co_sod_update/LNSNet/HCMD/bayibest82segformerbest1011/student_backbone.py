from thop import profile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from WBY_rail.IENet.bayibest82.baseapi.newapii711715  import BasicConv2d,node,CA,GM2  #atttention
# from second_model.IENet.mobilevit import mobilevit_s
from mmcv import Config
from mmseg.models import build_segmentor
from third.mix_transformer import mit_b0
from collections import OrderedDict
class AM2(nn.Module):
    def __init__(self, in_channel1, in_channel2, in_channel4, in_channel5):  #rgm1, rgm2, rlayer_features[1], rlayer_features[0]
        super(AM2, self).__init__()
        self.u1 = BasicConv2d(in_channel1, 96, 3, 1, 1)
        self.u2 = BasicConv2d(in_channel2, 96, 3, 1, 1)
        self.u4 = BasicConv2d(in_channel4, 96, 3, 1, 1)
        self.u42 = BasicConv2d(in_channel5, 96, 3, 1, 1)  ####84
        self.u5 = BasicConv2d(96 * 4, 96, 3, 1, 1)
        self.u6 = BasicConv2d(96, 96, 3, 1, 1)
        self.gcn1 = Gru(96,96)
        self.gcn2 = Gru(96, 96)
        self.gcn3 = Gru(96, 96)
        self.gcn4 = Gru(96, 96)
    def forward(self, x1, x2, x4, x5):
        gm1 = self.u1(x1)
        gm2 = self.u2(x2)
        r2 = self.u4(x4)
        r3 = self.u42(x5) ###84
        # br = F.interpolate(r2, scale_factor=2)
        br = r2
        gm1 = F.interpolate(gm1, size=96)
        br = F.interpolate(br, size=96)
        gm2 = F.interpolate(gm2, size= 96)
        r2 = F.interpolate(r2, size=96)
        r3 = F.interpolate(r3, size=96)
        # print("hhhhhbao")
        # print(gm1.size(),gm2.size(),br.size(),r3.size())
        # res = torch.cat((gm1, gm2, br), dim=1)
        # res = self.u6(self.u5(res) + gm1)
        res1 = self.gcn1(gm1,gm2)  ##yuanben douyong1
        # print(gm1.size(), br.size())
        res2 = self.gcn2(gm1, br)
        res3 = self.gcn3(gm2, br)
        res4 = self.gcn4(gm1,r3)
        # print("hhh")
        res = torch.cat((res1,res2,res3,res4), dim=1)
        res = self.u6(self.u5(res) + gm1)
        return res
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h



class GCN_MS(nn.Module):
    def __init__(self, num_state, num_node, bias=False):  #96 192
        super(GCN_MS, self).__init__()
        # print(num_node,num_state)
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

        self.max2 = nn.AdaptiveMaxPool2d((None,int(num_state / 2)))
        self.mc2 = nn.Conv1d(int(num_state / 2), int(num_state / 2), kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.mc2_1 = nn.Conv1d(int(num_state / 2), int(num_state), kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.max4 = nn.AdaptiveMaxPool2d((None, int(num_state / 4)))
        self.mc4 = nn.Conv1d(int(num_state / 2), int(num_state / 2), kernel_size=1, padding=0, stride=1, groups=1,
                             bias=True)
        self.mc4_1 = nn.Conv1d(int(num_state / 4), int(num_state), kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.max8 = nn.AdaptiveMaxPool2d((None, int(num_state / 8)))
        self.mc8 = nn.Conv1d(int(num_state / 2), int(num_state / 2), kernel_size=1, padding=0, stride=1, groups=1,
                             bias=True)
        self.mc8_1 = nn.Conv1d(int(num_state / 8), int(num_state), kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
    def forward(self, x):
        x = x.permute(0,2,1) # 96 * 192
        h1 = self.max2(x)
        h1_1 = self.mc2(h1) + h1
        h1_1 = h1_1.contiguous().permute(0,2,1)
        h1_1 = self.relu(h1_1)
        h1_1 = self.mc2_1(h1_1)

        h2 = self.max4(x)
        h2_1 = self.mc4(h2) + h2
        h2_1 = h2_1.contiguous().permute(0, 2, 1)
        h2_1 = self.relu(h2_1)
        h2_1 = self.mc4_1(h2_1)

        h3 = self.max8(x)
        h3_1 = self.mc8(h3) + h3
        h3_1 = h3_1.contiguous().permute(0, 2, 1)
        h3_1 = self.relu(h3_1)
        h3_1 = self.mc8_1(h3_1)
        return h1_1 + h2_1 + h3_1

class Gru(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(Gru, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = BasicConv2d(num_in, self.num_s,  kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        self.conv_state2 = BasicConv2d(num_in, self.num_s, kernel_size=kernel_size, padding=padding)
        # generate projection and inverse projection functions
        self.conv_proj2 = BasicConv2d(num_in, self.num_n, kernel_size=kernel_size, padding=padding)

        # reasoning by graph convolution
        self.gcn1 = GCN_MS(num_state=self.num_s, num_node=self.num_n)
        self.gcn2 = GCN_MS(num_state=self.num_s, num_node=self.num_n)
        # fusion
        self.fc_2 = nn.Conv2d(num_in, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1),
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in)


    def forward(self, x, y):
        batch_size = x.size(0)

        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1)
        y_proj_reshaped = self.conv_proj(y).view(batch_size, self.num_n, -1)

        x_state_2 = self.conv_state2(x).view(batch_size, self.num_s, -1)


        x_n_state1 = torch.bmm(x_state_reshaped, y_proj_reshaped.permute(0, 2, 1))
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))


        # print(x_n_state2.size(),"hhh")  192*96
        x_n_rel1 = self.gcn1(x_n_state2)
        x_n_rel2 = self.gcn2(x_n_rel1)

        x_state_reshaped = torch.bmm(x_n_rel2.permute(0,2,1), x_state_2)

        x_state = x_state_reshaped.view(batch_size, 96, 96,96)
        # print(x_state.size())
        # fusion
        out = x + self.blocker(self.fc_2(x_state)) + y
        # print(out.size())

        return out
class M(nn.Module):
    def load_pret(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)["state_dict"]
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.resnet.load_state_dict(new_state_dict3, strict=False)


    def __init__(self, mode='small'):
        super(M, self).__init__()
        # cfg_path = '/home/wby/PycharmProjects/Transformer_backbone/SegFormerguanfnag/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py'
        # cfg = Config.fromfile(cfg_path)
        # self.resnet = build_segmentor(cfg.model)
        self.resnet = mit_b0()
        self.am = AM2(32, 160, 32, 64)
        self.u4 = nn.Conv2d(32, 1, 1)
        self.gm1 = GM2(32, 64, 80)
        self.gm2 = GM2(160, 320, 20)

        self.am_d = AM2(32, 160, 32, 64)
        self.u4_d = nn.Conv2d(32, 1, 1)
        self.gm1_d = GM2(32, 64, 80)
        self.gm2_d = GM2(160, 320, 20)

        self.sup1 = BasicConv2d(320 * 2, 1, 3, 1, 1)
        self.sup2 = BasicConv2d(128 * 2, 1, 3, 1, 1)
        self.sup3 = BasicConv2d(64 * 2, 1, 3, 1, 1)

        # self.student = mobilevit_s()
        self.s1 = nn.Conv2d(128, 1, 1)

        self.bound1 = nn.Conv2d(64, 1, 1)
        self.bound2 = nn.Conv2d(320, 1, 1)
        self.bound3 = nn.Conv2d(64, 1, 1)
        self.bound4 = nn.Conv2d(320, 1, 1)

        self.node1 = node(32, 64, 80, 40) # 80, 40)
        self.node2 = node(160, 256, 20, 10) # 20, 10)

        self.gcn1 = Gru(64,64)
        self.gcn2 = Gru(64, 64)

        self.newrd1 = nn.Conv2d(32, 64, 1)
        self.newrd3 = nn.Conv2d(160, 320, 1)

        self.newr1 = nn.Conv2d(32, 64, 1)
        self.newr2 = nn.Conv2d(64, 128, 1)
        self.newr3 = nn.Conv2d(160, 320, 1)
        self.newr4 = nn.Conv2d(256, 512, 1)

        self.newd1 = nn.Conv2d(32, 64, 1)
        self.newd2 = nn.Conv2d(64, 128, 1)
        self.newd3 = nn.Conv2d(160, 320, 1)
        self.newd4 = nn.Conv2d(256, 512, 1)
        # self.u5 = nn.Conv2d(96*2, 1, 1)

        self.ablr1 = nn.Conv2d(64, 32, 1)
        self.ablr2 = nn.Conv2d(256, 160, 1)
        self.ablr3 = nn.Conv2d(160, 32, 1)
    def forward(self, r, d):

        temperature = 1
        rte = self.resnet.forward(r)
        # for i in rte:
        #     print(i.size())
        d = torch.cat((d, d, d), dim=1)
        # print(d.size())
        ###############################T###################################
        dte = self.resnet.forward(d)
        # print("dlayer_outputshhhh", dte[0].size(), dte[1].size(), dte[2].size(),
        #       dte[3].size())
        # for i in dte:
        #     print(i.size())
        # rd1, rd2, wr1, wd1 = self.node1(rte[0], rte[1], dte[0], dte[1],
        #                       temperature)
        # rd3, rd4, wr2, wd2 = self.node2(rte[2], rte[3], dte[2], dte[3],
        #                       temperature)
        rd1 = F.interpolate(self.ablr1(rte[1])+self.ablr1(dte[1]), scale_factor=2)+rte[0]+dte[0]
        rd3 = F.interpolate(self.ablr2(rte[3])+self.ablr2(dte[3]), scale_factor=2)+rte[2]+dte[2]

        # rgm1 = self.gm1(rd1, rte[0], temperature)
        # rgm2 = self.gm2(rd3, rte[2], temperature)
        rgm1 = rd1 + rte[0]
        rgm2 = rd3 + rte[2]

        # print(rgm1.size(), rgm2.size(), rte[0].size(), rte[1].size())
        # mres1 = self.am(rgm1, rgm2, rte[0], rte[1])
        # print(rgm1.size(), rgm2.size(), rte[0].size(), rte[1].size())
        mres1 = rgm1 + F.interpolate(self.ablr3(rgm2), scale_factor=4) + F.interpolate(self.ablr1(rte[1]), scale_factor=2) + rte[0]
        res1 = self.u4(mres1)
        res1 = F.interpolate(res1, 320)

        # tgm1 = self.gm1_d(rd1, dte[0], temperature)
        # tgm2 = self.gm2_d(rd3, dte[2], temperature)
        tgm1 = rd1 + dte[0]
        tgm2 = rd3 + dte[2]
        # mres2 = self.am_d(tgm1, tgm2, dte[0], dte[1])  # !!!!!!
        mres2 = tgm1 + F.interpolate(self.ablr3(tgm2), scale_factor=4) + F.interpolate(self.ablr1(dte[1]), scale_factor=2) + dte[0]
        res2 = self.u4_d(mres2)
        res2 = F.interpolate(res2, 320)

        res = res1 + res2

        newrd1 = self.newrd1(rd1)
        newrd3 = self.newrd3(rd3)
        # sr = F.interpolate(self.s1(self.student.forward(r)), size=384)
        newr = []
        i = self.newr1(rte[0])
        newr.append(i)
        i = self.newr2(rte[1])
        newr.append(i)
        i = self.newr3(rte[2])
        newr.append(i)
        i = self.newr4(rte[3])
        newr.append(i)
        newd = []
        i = self.newd1(dte[0])
        newd.append(i)
        i = self.newd2(dte[1])
        newd.append(i)
        i = self.newd3(dte[2])
        newd.append(i)
        i = self.newd4(dte[3])
        newd.append(i)
        # bound1 = F.interpolate(self.bound1(rgm1), size=320)
        # bound3 = F.interpolate(self.bound3(tgm1), size=320)
        # contour1 = F.interpolate(self.bound2(rgm2), size=320)
        # contour2 = F.interpolate(self.bound4(tgm2), size=320)  ####726
        if self.training:
            return res2, res1, res, newrd1, newrd3, newr, newd ####726
        else:
            return res2, res1, res
        # return res2, res1, res
if __name__=='__main__':
    a = torch.randn(1, 3, 224, 224).cuda()
    b = torch.randn(1, 1, 224, 224).cuda()
    net = M().cuda()
    flops, parameters = profile(net, (a, b))
    print(flops / 1e9, parameters / 1e6)
    # for i in out:
#         print(i.shape)
