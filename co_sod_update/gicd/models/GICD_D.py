import torch
from torch import nn
import torch.nn.functional as F
from Co_SOD.gicd.models.vgg import VGG_Backbone
import numpy as np
import torch.optim as optim
from torchvision.models import vgg16
from thop import profile
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class EnLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x


class LatLayer(nn.Module):
    def __init__(self, in_channel):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class DSLayer(nn.Module):
    def __init__(self, in_channel=64):
        super(DSLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.predlayer = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        x = self.enlayer(x)
        x = self.predlayer(x)
        return x


class GINet(nn.Module):
    def __init__(self, mode='train'):
        super(GINet, self).__init__()
        self.gradients = None
        self.backbone = VGG_Backbone()
        self.backbone_d = VGG_Backbone()

        self.toplayer = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0))

        self.latlayer4 = LatLayer(in_channel=512)
        self.latlayer3 = LatLayer(in_channel=256)
        self.latlayer2 = LatLayer(in_channel=128)
        self.latlayer1 = LatLayer(in_channel=64)

        self.enlayer4 = EnLayer()
        self.enlayer3 = EnLayer()
        self.enlayer2 = EnLayer()
        self.enlayer1 = EnLayer()

        self.dslayer4 = DSLayer()
        self.dslayer3 = DSLayer()
        self.dslayer2 = DSLayer()
        self.dslayer1 = DSLayer()

    def save_gradient(self, grad):
        self.gradients = grad

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def _fg_att(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = F.interpolate(pred,
                             size=(H, W),
                             mode='bilinear',
                             align_corners=True)
        return feat * pred

    def forward(self, x, y, co_coding):
        [_, _, H, W] = x.size()
        # with torch.no_grad():
        x1 = self.backbone.conv1(x)
        x2 = self.backbone.conv2(x1)
        x3 = self.backbone.conv3(x2)
        x4 = self.backbone.conv4(x3)

        d1 = self.backbone_d.conv1(y)
        d2 = self.backbone_d.conv2(d1)
        d3 = self.backbone_d.conv3(d2)
        d4 = self.backbone_d.conv4(d3)

        x5 = self.backbone.conv5(x4)
        d5 = self.backbone.conv5(d4)
        x1 = x1 + d1
        x2 = x2 + d2
        x3 = x3 + d3
        x4 = x4 + d4
        x5 = x5 + d5

        # x5.requires_grad_()
        # x5.register_hook(self.save_gradient)
        x5_p = self.backbone.avgpool(x5)
        _x5_p = x5_p.view(x5_p.size(0), -1)
        pred_vector = self.backbone.classifier(_x5_p)

        # co_coding = co_coding.requires_grad_()
        similarity = torch.sum(co_coding.cuda() * pred_vector)

        # similarity.backward(retain_graph=True)
        # cweight = F.adaptive_avg_pool2d(self.gradients, (1, 1))
        # cweight = F.relu(cweight)
        # cweight = (cweight - torch.min(cweight)) / (torch.max(cweight) -
        #                                             torch.min(cweight) + 1e-20)
        weighted_x5 = x5# * cweight
        cam = torch.mean(weighted_x5, dim=1).unsqueeze(1)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-6)
        cam = torch.clamp(cam, 0, 1)

        # with torch.no_grad():
        ########## Up-Sample ##########
        preds = []
        p5 = self.toplayer(weighted_x5)
        _pred = cam
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p4 = self._upsample_add(p5, self.latlayer4(self._fg_att(x4,
                                                                _pred)))
        p4 = self.enlayer4(p4)
        _pred = self.dslayer4(p4)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p3 = self._upsample_add(p4, self.latlayer3(self._fg_att(x3,
                                                                _pred)))
        p3 = self.enlayer3(p3)
        _pred = self.dslayer3(p3)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p2 = self._upsample_add(p3, self.latlayer2(self._fg_att(x2,
                                                                _pred)))
        p2 = self.enlayer2(p2)
        _pred = self.dslayer2(p2)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        p1 = self._upsample_add(p2, self.latlayer1(self._fg_att(x1,
                                                                _pred)))
        p1 = self.enlayer1(p1)
        _pred = self.dslayer1(p1)
        preds.append(
            F.interpolate(_pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        return preds


class GICD(nn.Module):
    def __init__(self, mode='train'):
        super(GICD, self).__init__()
        self.co_classifier = vgg16(pretrained=True).eval() # None
        self.co_classifier_d = vgg16(pretrained=True).eval() # None
        self.ginet = GINet()

    def forward(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        [_, N, _, _, _] = x.size()

        # with torch.no_grad():
        ######### Co-Classify ########
        co_coding = 0
        for inum in range(N):
            co_coding += self.co_classifier(
                x[:, inum, :, :, :]).cpu().data.numpy()
        co_coding = torch.from_numpy(co_coding)
        co_coding = F.softmax(co_coding, dim=1)  # 注意了 dim

        # with torch.no_grad():
        #     ######### Co-Classify ########
        #     co_coding = 0
        #     for inum in range(N):
        #         co_coding += self.co_classifier(
        #             x[:, inum, :, :, :]).cpu().data.numpy()
        #     co_coding = torch.from_numpy(co_coding)
        #     co_coding = F.softmax(co_coding, dim=1)  #注意了 dim

        ########## Co-SOD ############
        preds = []
        for inum in range(N):
            ipreds = self.ginet(x[:, inum, :, :, :],y[:, inum, :, :, :], co_coding)  # list  ([2, 1, 256, 256]) × 5
            # for i in ipreds:
            #     print("ipreds", i.size())
            preds.append(ipreds)
        # for i in preds[-1]:
        #     print("pred", i.size())
        return preds
from time import time
from tqdm import tqdm
if __name__ == '__main__':
    # a = torch.randn(2, 1, 3, 256, 256).cuda()
    # b = torch.randn(2, 1, 3, 256, 256).cuda()
    a = torch.randn(2, 3, 256, 256).cuda()
    b = torch.randn(2, 3, 256, 256).cuda()
    c = torch.randn(5, 1, 256, 256).cuda()
    model = GICD()
    model.cuda()
    out = model(a, b)
    rgb = torch.randn(a.size()).cuda()
    dep = torch.randn(b.size()).cuda()
    flops, params = profile(model, (a,b))

    print('Flops', flops / 1e9, 'G')
    print('Params: ', params / 1e6, 'M')


    def print_network(model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print("The number of parameters:{}M".format(num_params / 1000000))


    for i in tqdm(range(100)):
        # warm up
        p = model(a, b)
        # p = model(x)
        # p = p + 1

    total_t = 0

    for i in tqdm(range(200)):
        start = time()
        p = model(a, b)
        # p = model(x)
        # p = p + 1 # replace torch.cuda.synchronize()
        total_t += time() - start

    print("origin batch 1 FPS", 400 / total_t)
    print("batch every seconds,", total_t / 400)
