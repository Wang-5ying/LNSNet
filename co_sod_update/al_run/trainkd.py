from random import seed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
from thop import profile

from COA_RGBD_SOD.al.models.New_Test_Shunted_T import RGBD_sal
from COA_RGBD_SOD.al.models.New_Test_Shunted_B import RGBD_teacher_sal

from COA_RGBD_SOD.al.util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from COA_RGBD_SOD.al.dataset import get_loader
import torchvision.utils as vutils
import torch.nn.functional as F
# import pytorch_toolbelt.losses as PTL
from COA_RGBD_SOD.al.config import Config
from COA_RGBD_SOD.al.loss import saliency_structure_consistency, DSLoss
from COA_RGBD_SOD.al.util import generate_smoothed_gt
from COA_RGBD_SOD.al.evaluation.dataloader import EvalDataset
from COA_RGBD_SOD.al.evaluation.evaluator import Eval_thread

from COA_RGBD_SOD.al.Loss.kd_losses.ofd import OFD
from COA_RGBD_SOD.al.Loss.kd_losses.sp import *
from COA_RGBD_SOD.al.Loss.losses import *


# model

from COA_RGBD_SOD.al.pytorch_iou.__init__ import IOU


UP2 = nn.UpsamplingBilinear2d(scale_factor=2)
UP4 = nn.UpsamplingBilinear2d(scale_factor=4)
UP8 = nn.UpsamplingBilinear2d(scale_factor=8)
UP16 = nn.UpsamplingBilinear2d(scale_factor=16)
UP32 = nn.UpsamplingBilinear2d(scale_factor=32)


def dice_loss(pred, mask):
    mask = mask
    pred = pred
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            # shape = x.shape[-2:]
            # print(shape)
            # print(h)
            y = F.interpolate(y, (h, h), mode="bilinear")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output
        y = self.conv2(x)
        return y

def SEM(frgb, fdepth):
    Aggreation = frgb * fdepth
    sub = frgb - fdepth
    f = Aggreation + sub
    return f

def CEM(frgb, fdepth):
    Aggreation = frgb * fdepth
    ADD = frgb + fdepth
    f = Aggreation + ADD
    return f

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all


# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='M',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=320, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='DUTS_class',  # default='Jigsaw2_DUTS',
                    type=str,
                    help="Options: 'DUTS_class'")
parser.add_argument('--size',
                    default=256,
                    type=int,
                    help='input size')
parser.add_argument('--ckpt_dir', default="/home/map/Alchemist/COA/COA_RGBD_SOD/ckpt/Ablation_KD/New_Test_Shunted_T_Ablation_F/Pth",
                    help='Temporary folder')

parser.add_argument('--testsets',
                    default='val5.0',
                    # default='', CoCA+CoSOD3k+Cosal2015  val5.0 RGBD_CoSeg183  RGBD_CoSal1k RGBD_CoSal150
                    type=str,
                    help="Options: 'CoCA','Cosal2015','CoSOD3k','iCoseg','MSRC'")

parser.add_argument('--val_dir',
                    default='tmp4val',
                    type=str,
                    help="Dir for saving tmp results for validation.")

args = parser.parse_args()

config = Config()

# Prepare dataset
if args.trainset == 'DUTS_class':
    root_dir = '/home/map/Alchemist/COA/data/'
    train_img_path = os.path.join(root_dir, 'images/DUTS_class')
    train_gt_path = os.path.join(root_dir, 'gts/DUTS_class')
    train_depth_path = os.path.join(root_dir, 'depths/DUTS_class')
    train_edge_path = os.path.join(root_dir, 'edge/DUTS_class')
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              train_depth_path,
                              train_edge_path,
                              args.size,
                              1,
                              max_num=config.batch_size,
                              istrain=True,
                              shuffle=False,
                              num_workers=8,
                              pin=True)
    train_img_path_seg = os.path.join(root_dir, 'images/coco-seg')
    train_gt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    train_depth_path_seg = os.path.join(root_dir, 'depths/coco-seg')
    train_edge_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    train_loader_seg = get_loader(
        train_img_path_seg,
        train_gt_path_seg,
        train_depth_path_seg,
        train_edge_path_seg,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )
elif args.trainset == 'CoCA':
    root_dir = '/home/map/Alchemist/COA/data/'
    train_img_path = os.path.join(root_dir, 'images/CoCA')
    train_gt_path = os.path.join(root_dir, 'gts/CoCA')
    train_depth_path = os.path.join(root_dir, 'depths/CoCA')
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              train_depth_path,
                              args.size,
                              1,
                              max_num=config.batch_size,
                              istrain=True,
                              shuffle=False,
                              num_workers=8,
                              pin=True)
    train_img_path_seg = os.path.join(root_dir, 'images/coco-seg')
    train_gt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    train_depth_path_seg = os.path.join(root_dir, 'depths/coco-seg')
    train_loader_seg = get_loader(
        train_img_path_seg,
        train_gt_path_seg,
        train_depth_path_seg,
        args.size,
        1,
        max_num=config.batch_size,
        istrain=True,
        shuffle=True,
        num_workers=8,
        pin=True
    )
else:
    print('Unkonwn train dataset')
    print(args.dataset)

test_loaders = {}
for testset in args.testsets.split('+'):
    test_loader = get_loader(
        os.path.join('/home/map/Alchemist/COA/data', 'images', testset),
        os.path.join('/home/map/Alchemist/COA/data', 'gts', testset),
        os.path.join('/home/map/Alchemist/COA/data', 'depths', testset),
        os.path.join('/home/map/Alchemist/COA/data', 'gts', testset),
        args.size, 1, max_num=config.batch_size, istrain=False, shuffle=False, num_workers=8, pin=True
    )
    test_loaders[testset] = test_loader

if config.rand_seed:
    set_seed(config.rand_seed)

# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_file = os.path.join(args.ckpt_dir, "log_loss.txt")
logger_loss_idx = 1

# Init model
device = torch.device("cuda")
model = RGBD_sal()
model_T = RGBD_teacher_sal()
model.load_pre('/home/map/Alchemist/COA/COA_RGBD_SOD/Shunted/ckpt_T.pth')

base_weights = torch.load('/home/map/Alchemist/COA/COA_RGBD_SOD/ckpt/Teacher_New_Test_Shunted_B_ep128_Smeasure0.8778.pth')
new_state_dict = OrderedDict()
for k, v in base_weights.items():
    name = k  # remove 'module.'
    new_state_dict[name] = v
model_T.load_state_dict(new_state_dict)
model_T.eval()


model_T = model_T.to(device)
model = model.to(device)
if config.lambda_adv:
    from COA_RGBD_SOD.al.adv import Discriminator

    disc = Discriminator(channels=1, img_size=args.size).to(device)
    optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.9, 0.99])
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()

CE = torch.nn.BCEWithLogitsLoss().cuda()
IOU = IOU(size_average=True).cuda()
ABF1 = ABF(64, 128, 64, fuse=True).cuda()
ABF2 = ABF(128, 256, 128, fuse=True).cuda()
ABF3 = ABF(256, 512, 256, fuse=True).cuda()
ABF4 = ABF(512, 512, 512, fuse=False).cuda()

KLDLoss1 = KLDLoss().cuda()
KLDLoss2 = KLDLoss().cuda()
KLDLoss3 = KLDLoss().cuda()
KLDLoss4 = KLDLoss().cuda()
KLDLoss5 = KLDLoss().cuda()
KLDLoss6 = KLDLoss().cuda()
KLDLoss7 = KLDLoss().cuda()
KLDLoss8 = KLDLoss().cuda()



IFVDLoss = IFVDLoss().cuda()
OFDLoss1 = OFD(in_channels=64, out_channels=64).cuda()
OFDLoss2 = OFD(in_channels=128, out_channels=128).cuda()
OFDLoss3 = OFD(in_channels=256, out_channels=256).cuda()
OFDLoss4 = OFD(in_channels=512, out_channels=512).cuda()

SP_Loss = SP().cuda()
AT_Loss = ATLoss().cuda()


all_params = model.parameters()
# Setting optimizer
optimizer = optim.Adam(params=all_params, lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step_size, gamma=0.1)

# Why freeze the backbone?...
if config.freeze:
    for key, value in model.named_parameters():
        if 'bb' in key and 'bb.conv5.conv5_3' not in key:
            value.requires_grad = False

# log model and optimizer params
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
dsloss = DSLoss()


def main():
    val_measures = []
    global temperature
    temperature = 34
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model.load_state_dict(torch.load(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch, temperature)
        print('temperature:', temperature)
        if temperature != 1:
            temperature -= 3
            print('Change temperature to:', str(temperature))
        if config.validation:
            measures = validate(model, test_loaders, args.testsets, temperature)
            val_measures.append(measures)
            print(
                'Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                    epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                    np.max(np.array(val_measures)[:, 0]))
            )
            # print(
            #     'Validation: MAE on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with MAE {:.4f}'.format(
            #         epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
            #         np.max(np.array(val_measures)[:, 0]))
            # )
        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            path=args.ckpt_dir)
        # if epoch >= args.epochs - config.val_last:

        # if epoch >= args.epochs - 50:
        if epoch >= args.epochs - 8 or measures[0] >= 0.8600:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'Shunted_Simple_kd_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))

        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in
                                       os.listdir(args.ckpt_dir) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'Shunted_Simple_kd_best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))





def train(epoch, temperature):
    loss_log = AverageMeter()
    loss_log_triplet = AverageMeter()
    global logger_loss_idx
    model.train()
    model_T.eval()

    # for batch_idx, (batch, batch_seg) in enumerate(zip(train_loader, train_loader_seg)):
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        depths = batch[2].to(device).squeeze(0)

        # edges = batch[3].to(device).squeeze(0)
        # cls_gts = torch.LongTensor(batch[-1]).to(device)
        # print("gt--depth",gts.size(),edges.size())
        # print(cls_gts)



        with torch.no_grad():

            T_S_1, T_S_2, T_S_3, T_S_4, T_F_1, T_F_2, T_F_3, T_F_4, T_R_4, \
            T_D_4, T_r_1, T_r_2, T_r_3, T_r_4, T_d_1, T_d_2, T_d_3, T_d_4 = model_T(inputs, depths)



        S_S_1, S_S_2, S_S_3, S_S_4, S_F_1, S_F_2, S_F_3, S_F_4, S_R_4, \
        S_D_4, S_r_1, S_r_2, S_r_3, S_r_4, S_d_1, S_d_2, S_d_3, S_d_4 = model(inputs, depths)



        # up_sum = torch.cat([T_S_1, T_S_2, T_S_3, T_S_4], dim=1)
        # socre_s = F.softmax(up_sum, dim=1)
        # a, b, c, d = socre_s.split(1, dim=1)
        #
        # F_T_1 = F.adaptive_max_pool2d(T_F_1, (1, 1))
        # F_T_2 = F.adaptive_max_pool2d(T_F_2, (1, 1))
        # F_T_3 = F.adaptive_max_pool2d(T_F_3, (1, 1))
        # F_T_4 = F.adaptive_max_pool2d(T_F_4, (1, 1))
        # F_sum = torch.cat([F_T_1, F_T_2, F_T_3, F_T_4], dim=1)
        # socre_F = F.softmax(F_sum, dim=1)
        # F_a, F_b, F_c, F_d = socre_F.split([64, 128, 256, 512], dim=1)

        # UP_T_R_4 = UP8(T_R_4)
        # UP_T_D_4 = UP8(T_D_4)
        # UP_S_R_4 = UP8(S_R_4)
        # UP_S_D_4 = UP8(S_D_4)
        #
        # s_s_1 = F.sigmoid(S_S_1)
        # s_s_2 = F.sigmoid(S_S_2)
        # s_s_3 = F.sigmoid(S_S_3)
        # s_s_4 = F.sigmoid(S_S_4)
        #
        # t_s_1 = F.sigmoid(T_S_1)
        # t_s_2 = F.sigmoid(T_S_2)
        # t_s_3 = F.sigmoid(T_S_3)
        # t_s_4 = F.sigmoid(T_S_4)
        #
        # s_s_1 = 1 - s_s_1
        # s_s_2 = 1 - s_s_2
        # s_s_3 = 1 - s_s_3
        # s_s_4 = 1 - s_s_4
        #
        # t_s_1 = 1 - t_s_1
        # t_s_2 = 1 - t_s_2
        # t_s_3 = 1 - t_s_3
        # t_s_4 = 1 - t_s_4


        loss = 0

        if epoch <= 30:

            # UP_T_r_4 = UP8(T_r_4)
            # UP_T_d_4 = UP8(T_d_4)
            # UP_S_r_4 = UP8(S_r_4)
            # UP_S_d_4 = UP8(S_d_4)

            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = CE(S_S_3, gts)
            loss8 = CE(S_S_4, gts)
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            # los1_8 = KLDLoss1(S_S_1 / 4, T_S_1 / 4, gts, 5)
            # los2_8 = KLDLoss2(S_S_2 / 4, T_S_2 / 4, gts, 5)
            # los3_8 = KLDLoss3(S_S_3 / 4, T_S_3 / 4, gts, 5)
            # los4_8 = KLDLoss4(S_S_4 / 4, T_S_4 / 4, gts, 5)
            # loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            # loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt
            loss_kd = 0

            # loss_kd_b1 = KLDLoss1(s_s_1 / temperature, t_s_1 / temperature, gts, 4)
            # loss_kd_b2 = KLDLoss1(s_s_2 / temperature, t_s_2 / temperature, gts, 4)
            # loss_kd_b3 = KLDLoss1(s_s_3 / temperature, t_s_3 / temperature, gts, 4)
            # loss_kd_b4 = KLDLoss1(s_s_4 / temperature, t_s_4 / temperature, gts, 4)
            # loss_kd_b = loss_kd_b1 * 4 + loss_kd_b2 + loss_kd_b3 + loss_kd_b4
            loss_kd_b = 0

            # loss_encoder_r4 = SP_Loss(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            # loss_encoder_d4 = SP_Loss(S_D_4 / 4, T_D_4 / 4).cuda() * 4
            # up_loss_encoder_r4 = SP_Loss(UP_S_R_4 / 4, UP_T_R_4 / 4).cuda() * 4
            # up_loss_encoder_d4 = SP_Loss(UP_S_D_4 / 4, UP_T_D_4 / 4).cuda() * 4
            # loss_decoder_r1 = SP_Loss(S_r_1 / temperature, T_r_1 / temperature).cuda()
            # loss_decoder_r2 = SP_Loss(S_r_2 / temperature, T_r_2 / temperature).cuda()
            # loss_decoder_r3 = SP_Loss(S_r_3 / temperature, T_r_3 / temperature).cuda()
            # loss_decoder_r4 = SP_Loss(S_r_4 / temperature, T_r_4 / temperature).cuda()
            # up_loss_decoder_r4 = SP_Loss(UP_S_r_4 / temperature, UP_T_r_4 / temperature).cuda()
            # loss_decoder_d1 = SP_Loss(S_d_1 / temperature, T_d_1 / temperature).cuda()
            # loss_decoder_d2 = SP_Loss(S_d_2 / temperature, T_d_2 / temperature).cuda()
            # loss_decoder_d3 = SP_Loss(S_d_3 / temperature, T_d_3 / temperature).cuda()
            # loss_decoder_d4 = SP_Loss(S_d_4 / temperature, T_d_4 / temperature).cuda()
            # up_loss_decoder_d4 = SP_Loss(UP_S_d_4 / temperature, UP_T_d_4 / temperature).cuda()
            #
            # loss_encoder = loss_encoder_r4 + loss_encoder_d4 + loss_decoder_r1 + loss_decoder_r2 + loss_decoder_r3 + \
            #                loss_decoder_r4 + loss_decoder_d1 + loss_decoder_d2 + loss_decoder_d3 + loss_decoder_d4 + \
            #                up_loss_encoder_r4 + up_loss_encoder_d4 + up_loss_decoder_r4 + up_loss_decoder_d4

            loss_encoder = 0

            loss_F_1 = AT_Loss(S_F_1 / temperature, T_F_1 / temperature, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / temperature, T_F_2 / temperature, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / temperature, T_F_3 / temperature, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / temperature, T_F_4 / temperature, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4
            loss_midel = loss_midel

        elif 30 < epoch <=60:
            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = 0
            loss8 = 0
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            # los1_8 = KLDLoss1(S_S_1 / 1, T_S_1 / 1, gts, 5)
            # los2_8 = KLDLoss2(S_S_2 / 1, T_S_2 / 1, gts, 5)
            # los3_8 = KLDLoss3(S_S_3 / 1, T_S_3 / 1, gts, 5)
            # los4_8 = KLDLoss4(S_S_4 / 1, T_S_4 / 1, gts, 5)
            #
            # loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            # loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt

            loss_kd = 0

            # loss_kd_b1 = KLDLoss1(s_s_1 / 4, t_s_1 / 4, gts, 4)
            # loss_kd_b2 = KLDLoss1(s_s_2 / 4, t_s_2 / 4, gts, 4)
            # loss_kd_b3 = KLDLoss1(s_s_3 / 4, t_s_3 / 4, gts, 4)
            # loss_kd_b4 = KLDLoss1(s_s_4 / 4, t_s_4 / 4, gts, 4)
            # loss_kd_b = loss_kd_b1 * 4 + loss_kd_b2 + loss_kd_b3 + loss_kd_b4
            loss_kd_b = 0

            # loss_encoder_r4 = SP_Loss(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            # loss_encoder_d4 = SP_Loss(S_D_4 / 4, T_D_4 / 4).cuda() * 4
            # up_loss_encoder_r4 = SP_Loss(UP_S_R_4 / 4, UP_T_R_4 / 4).cuda() * 4
            # up_loss_encoder_d4 = SP_Loss(UP_S_D_4 / 4, UP_T_D_4 / 4).cuda() * 4
            #
            # loss_encoder = loss_encoder_r4 + loss_encoder_d4 + up_loss_encoder_r4 + up_loss_encoder_d4
            loss_encoder = 0

            loss_F_1 = AT_Loss(S_F_1 / 1, T_F_1 / 1, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / 1, T_F_2 / 1, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / 1, T_F_3 / 1, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / 1, T_F_4 / 1, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4
            loss_midel = loss_midel

        elif 60 < epoch <= 90:
            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = 0
            loss8 = 0
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            # los1_8 = KLDLoss1(S_S_1 / temperature, T_S_1 / temperature, gts, 5)
            # los2_8 = KLDLoss2(S_S_2 / temperature, T_S_2 / temperature, gts, 5)
            # los3_8 = KLDLoss3(S_S_3 / temperature, T_S_3 / temperature, gts, 5)
            # los4_8 = KLDLoss4(S_S_4 / temperature, T_S_4 / temperature, gts, 5)
            # loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            # loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt

            loss_kd = 0

            loss_kd_b = 0

            # loss_encoder_r4 = SP_Loss(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            # loss_encoder_d4 = SP_Loss(S_D_4 / 4, T_D_4 / 4).cuda() * 4
            # up_loss_encoder_r4 = SP_Loss(UP_S_R_4 / 4, UP_T_R_4 / 4).cuda() * 4
            # up_loss_encoder_d4 = SP_Loss(UP_S_D_4 / 4, UP_T_D_4 / 4).cuda() * 4
            # loss_encoder = loss_encoder_r4 + loss_encoder_d4 + up_loss_encoder_r4 + up_loss_encoder_d4
            loss_encoder = 0

            loss_F_1 = AT_Loss(S_F_1 / temperature, T_F_1 / temperature, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / temperature, T_F_2 / temperature, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / temperature, T_F_3 / temperature, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / temperature, T_F_4 / temperature, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4
            loss_midel = loss_midel


        else:
            loss5 = CE(S_S_1, gts)
            loss6 = CE(S_S_2, gts)
            loss7 = 0
            loss8 = 0
            loss_gt = loss5 * 4 + loss6 + loss7 + loss8

            # los1_8 = KLDLoss1(S_S_1 / temperature, T_S_1 / temperature, gts, 5)
            # los2_8 = KLDLoss2(S_S_2 / temperature, T_S_2 / temperature, gts, 5)
            # los3_8 = KLDLoss3(S_S_3 / temperature, T_S_3 / temperature, gts, 5)
            # los4_8 = KLDLoss4(S_S_4 / temperature, T_S_4 / temperature, gts, 5)
            # loss_final = los1_8 * 4 + los2_8 + los3_8 + los4_8
            # loss_kd = loss_final * (1 - 0.0625) + 0.0625 * loss_gt
            loss_kd = 0

            loss_kd_b = 0

            # loss_encoder_r4 = SP_Loss(S_R_4 / 4, T_R_4 / 4).cuda() * 4
            # loss_encoder_d4 = SP_Loss(S_D_4 / 4, T_D_4 / 4).cuda() * 4
            # loss_encoder = loss_encoder_r4 + loss_encoder_d4
            loss_encoder = 0

            loss_F_1 = AT_Loss(S_F_1 / temperature, T_F_1 / temperature, gts, 4)
            loss_F_2 = AT_Loss(S_F_2 / temperature, T_F_2 / temperature, gts, 4)
            loss_F_3 = AT_Loss(S_F_3 / temperature, T_F_3 / temperature, gts, 4)
            loss_F_4 = AT_Loss(S_F_4 / temperature, T_F_4 / temperature, gts, 4)
            loss_midel = loss_F_1 * 4 + loss_F_2 + loss_F_3 + loss_F_4
            loss_midel = loss_midel

        loss_sal = loss_kd * 1 + loss_gt * 1 + 1 * loss_midel + loss_encoder * 1 + loss_kd_b

        loss_sal = loss_sal * 1

        loss += loss_sal
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Logger
        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}]'.format(epoch, args.epochs, batch_idx, len(train_loader))
            info_loss = 'Train Loss: loss_sal: {:.3f}'.format(loss_sal)
            logger.info(''.join((info_progress, info_loss)))
    # 对学习率进行更新
    scheduler.step()
    info_loss = '@==Final== Epoch[{0}/{1}]  Train Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=loss_log)
    if config.lambdas_sal_last['triplet']:
        info_loss += 'Triplet Loss: {loss.avg:.3f}  '.format(loss=loss_log_triplet)
    logger.info(info_loss)

    return loss_log.avg


def validate(model, test_loaders, testsets, temperature):
    model.eval()

    testsets = testsets.split('+')
    measures = []
    for testset in testsets[:1]:
        print('Validating {}...'.format(testset))
        test_loader = test_loaders[testset]

        saved_root = os.path.join(args.val_dir, testset)

        for batch in test_loader:

            inputs = batch[0].to(device).squeeze(0)
            # print(inputs.shape)
            gts = batch[1].to(device).squeeze(0)
            depths = batch[2].to(device).squeeze(0)
            edges = batch[3].to(device).squeeze(0)
            subpaths = batch[4]
            ori_sizes = batch[5]
            with torch.no_grad():
                scaled_preds = model(inputs, depths)[0]
                # scaled_preds = model(inputs, temperature)[0]
                # print("hhhh",scaled_preds.size())
            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                else:
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True).sigmoid()
                save_tensor_img(res, os.path.join(saved_root, subpath))

        eval_loader = EvalDataset(
            saved_root,  # preds
            os.path.join('/home/map/Alchemist/COA/data/gts', testset)  # GT
        )
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        if s_measure > config.val_measures['Smeasure']['val_38'] and 0:  # CoCA
            print("llll")
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)
    model.train()
    return measures


if __name__ == '__main__':
    main()
