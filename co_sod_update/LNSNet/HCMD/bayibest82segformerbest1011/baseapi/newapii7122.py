import torch
from torch import nn
import torch.nn.functional as F
from Dynamicconvolution.Dynamic.dynamic_conv import Dynamic_conv2d
class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.maxpool = nn.AdaptiveMaxPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
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
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x, temperature):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/temperature, 1)

class BM2(nn.Module):
    def __init__(self,in_channel1,in_channel2,size):
        super(BM2, self).__init__()
        # self.u1 = nn.Conv2d(in_channel2, in_channel1, 1)
        # self.u2 = nn.Conv2d(in_channel1, in_channel1, 1)
        # self.dl1 = nn.Conv2d(in_channel1 * 2, in_channel1, 3, 1, 1)
        self.u1 = Dynamic_conv2d(in_channel2, in_channel1, 1, 1, 1)
        self.u2 = Dynamic_conv2d(in_channel1, in_channel1, 1, 1, 1)
        self.dl1 = Dynamic_conv2d(in_channel1*2, in_channel1, 1, 1, 1)
        self.dca1 = CA(in_channel1*2)
        self.size = size
    def forward(self,x2, x3, temperature):
        x2 = self.u2(x2,temperature)
        l = self.u1(x3,temperature)
        l = F.interpolate(l, self.size)
        lr = torch.cat((x2, l), dim=1)
        b = self.dca1(lr)
        b = self.dl1(b,temperature)
        return b

class AM2(nn.Module):
    def __init__(self,in_channel1,in_channel2,in_channel3,in_channel4):
        super(AM2, self).__init__()
        # self.u1 = nn.Conv2d(in_channel1, 96, 1)
        # self.u2 = nn.Conv2d(in_channel2, 96, 1)
        # self.u3 = nn.Conv2d(in_channel3, 96, 1)
        # self.u4 = nn.Conv2d(in_channel4, 96, 1)
        # self.u5 = nn.Conv2d(96, 96, 1)
        self.u1 = Dynamic_conv2d(in_channel1, 96, 1, 1, 1)
        self.u2 = Dynamic_conv2d(in_channel2, 96, 1, 1, 1)
        self.u3 = Dynamic_conv2d(in_channel3, 96, 1, 1, 1)
        self.u4 = Dynamic_conv2d(in_channel4, 96, 1, 1, 1)
        self.u5 = Dynamic_conv2d(96, 96, 1, 1, 1)
    def forward(self,x1,x2,x3,x4):
        # print("712")
        gm1 = self.u1(x1)
        gm2 = self.u2(x2)
        b = self.u3(x3)
        r2 = self.u4(x4)
        br = F.interpolate(torch.mul(r2,b),scale_factor=2)
        gm12 = gm1 * F.interpolate(gm2,size=96)
        res = br*gm12 + gm12
        res = self.u5(res)
        return res

class GM2(nn.Module):
    def __init__(self,in_channel1,in_channel2,size):
        super(GM2, self).__init__()
        # self.u1 = nn.Conv2d(in_channel1, in_channel2, 1)
        # self.u2 = nn.Conv2d(in_channel2, in_channel2, 1)
        # self.dl1 = nn.Conv2d(in_channel1, in_channel2, 3, 1, 1)
        self.u1 = Dynamic_conv2d(in_channel1, in_channel2, 1, 1, 1)
        self.u2 = Dynamic_conv2d(in_channel2, in_channel2, 1, 1, 1)
        self.dl1 = Dynamic_conv2d(in_channel1, in_channel2, 1, 1, 1)
        self.dca1 = CA(in_channel2*2)
        self.sa1 = SA(kernel_size=7)
        self.size = size
    def forward(self, x1, x2):
        # print("11111")
        x1u = self.u1(x1)
        x2 = self.u2(x2)
        res1 = F.interpolate(x1u, self.size)
        # print("hhh",res1.size(),x2.size())
        res1 = res1 + x2
        res1h = self.sa1(res1)
        rd4l = F.interpolate(x1, self.size)
        rd4l = self.dl1(rd4l)
        res1 = torch.cat((res1, rd4l), dim=1)
        res1 = self.dca1(res1)
        res = res1 * res1h
        return res


class node(nn.Module):
    def __init__(self,in_channel1,in_channel2,size1,size2):
        super(node,self).__init__()
        self.ca1new = attention2d(int(in_channel1+in_channel2/4), 0.25, K=int(in_channel1+in_channel2/4),temperature=34)
        self.sa = SA(kernel_size=7)
        self.conv1 = BasicConv2d(int(in_channel1+in_channel2/4), in_channel1, 3, 1, 1)

        self.sa1 = SA(kernel_size=7)
        self.y1c = BasicConv2d(in_channel1, 1, 3, 1, 1)
        self.y1l = BasicConv2d(in_channel1, in_channel2, 3, 1, 1)

        self.yuc = BasicConv2d(in_channel2, in_channel1, 3, 1, 1)
        self.size1 = size1
        self.size2 = size2
        self.d_ls = int(self.size1/self.size2)

    def forward(self,r1,r2,d1,d2):
        #######r#############
        # rlayer_featuresx = F.interpolate(r2, self.size1, mode='bilinear', align_corners=True)
        rlayer_featuresx = F.pixel_shuffle(r2,self.d_ls)
        x12 = torch.cat((r1, rlayer_featuresx), dim=1)
        x12c = self.ca1new(x12).unsqueeze(-1).unsqueeze(-1)
        x12c = x12c * x12
        x12c = self.conv1(x12c)

        y2 = self.sa1(d2)
        y1 = F.interpolate(d1, self.size2, mode='bilinear', align_corners=True)
        y1 = self.y1c(y1)

        y12 = F.interpolate(torch.mul(y1, y2),scale_factor=2)
        y1l = self.y1l(d1)

        y = y1l * y12
        rd2 = F.interpolate(y, self.size2, mode='bilinear', align_corners=True)
        yu = self.yuc(y)
        rd1 = yu * x12c
        return rd1,rd2

class CA(nn.Module):
    def __init__(self,in_ch):
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
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, size= None, act_layer=nn.GELU, drop=0.):
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
        x= self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x