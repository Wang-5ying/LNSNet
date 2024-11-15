from random import seed
import torch
import torch.nn as nn
import torch.optim as optim
# from COA_RGBD_SOD.al.models.Second_model.Wave_CoNet_L_1 import CoNet as Teacher_Model_1
from COA_RGBD_SOD.al.models.Second_model.Wave_CoNet_T_WO_RFEM import CoNet as Student_Model
from COA_RGBD_SOD.al.models.Second_model.Wave_CoNet_L import CoNet as Teacher_Model
from COA_RGBD_SOD.al.util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from COA_RGBD_SOD.al.dataset import get_loader
import torch.nn.functional as F
from COA_RGBD_SOD.al.config import Config
from COA_RGBD_SOD.al.loss import saliency_structure_consistency, DSLoss
from COA_RGBD_SOD.al.evaluation.dataloader import EvalDataset
from COA_RGBD_SOD.al.evaluation.evaluator import Eval_thread
from COA_RGBD_SOD.al.Loss.kd_losses.sp import SP
from COA_RGBD_SOD.al.Loss.losses import *
from COA_RGBD_SOD.al.Loss.BDCLoss import MPSDLoss, CDLoss

from COA_RGBD_SOD.al.pytorch_iou.__init__ import IOU

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
parser.add_argument('--ckpt_dir',
                    default="/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/Alchemist/COA/COA_RGBD_SOD/ckpt/Wave_CoNet_T_WO_RFEM/pre_Pth1/",
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
model_T = Teacher_Model().to(device)
model = Student_Model()
# model_T1 = Teacher_Model_1().cuda()
model.load_pre('/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/Backbone_pth/release/ImageNet1K/p2t_tiny.pth')

# base_weights = torch.load('/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/Backbone_pth/best_ep124_Smeasure0.8660.pth')
# new_state_dict = OrderedDict()
# for k, v in base_weights.items():
#     name = k  # remove 'module.'
#     new_state_dict[name] = v
# model.load_state_dict(new_state_dict)

base_weights = torch.load('/home/map/Alchemist/COA/COA_RGBD_SOD/ckpt/Teacher_Wave_CoNet_L_ep76_Smeasure0.8757.pth')
new_state_dict = OrderedDict()
for k, v in base_weights.items():
    name = k  # remove 'module.'
    new_state_dict[name] = v
model_T.load_state_dict(new_state_dict)
model_T.eval()

model = model.to(device)
if config.lambda_adv:
    from COA_RGBD_SOD.al.adv import Discriminator

    disc = Discriminator(channels=1, img_size=args.size).to(device)
    optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.9, 0.99])
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()



class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

teachers_channels = [64, 128, 320, 640]
students_channels = [48, 96, 240, 384]



criterion1 = BCELOSS().cuda()

CE = torch.nn.BCEWithLogitsLoss().cuda()
IOU = IOU(size_average=True).cuda()
SP_Loss = SP().cuda()
KLD_Loss = KLDLoss().cuda()
CL = nn.TripletMarginWithDistanceLoss().cuda()
pool_ratios = [[12, 16, 20, 24], [6, 8, 10, 12], [3, 4, 5, 6], [1, 2, 3, 4]]
MPSD_Loss1 = MPSDLoss(students_channels[0], teachers_channels[0], pool_ratios[0], 1).cuda()
MPSD_Loss2 = MPSDLoss(students_channels[1], teachers_channels[1], pool_ratios[1], 2).cuda()
MPSD_Loss3 = MPSDLoss(students_channels[2], teachers_channels[2], pool_ratios[2], 4).cuda()
MPSD_Loss4 = MPSDLoss(students_channels[3], teachers_channels[3], pool_ratios[3], 8).cuda()

CD_Loss = CDLoss([48, 96, 240], [64, 128, 320]).cuda()
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
            temperature -= 1
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
        if epoch >= args.epochs - 8 or measures[0] >= 0.8560:
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, 'KD_Wave_CoNet_T_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))

        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in
                                       os.listdir(args.ckpt_dir) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'KD_Wave_CoNet_T_best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))


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
        background = 1 - gts
        with torch.no_grad():
            T_F, T_F1, T_f1, T_f2, T_f3, T_f4, T_a1, T_a2, T_a3, T_a4, T_fusion1, T_fusion2, T_fusion3 = model_T(inputs, depths)
            # T1_F, T1_F1, T1_f1, T1_f2, T1_f3, T1_f4, T1_a1, T1_a2, T1_a3, T1_a4 = model_T1(inputs, depths)

        S_F, S_F1, S_f1, S_f2, S_f3, S_f4, S_a1, S_a2, S_a3, S_a4, S_fusion1, S_fusion2, S_fusion3 = model(inputs, depths)

        S_Fb = 1 - F.sigmoid(S_F)
        S_F1b = 1 - F.sigmoid(S_F1)


        T_Fb  = 1 - F.sigmoid(T_F)
        T_F1b = 1 - F.sigmoid(T_F1)

        loss = 0

        if epoch <= 30:
            loss1 = CE(S_F, gts) + IOU(F.sigmoid(S_F), gts)
            loss2 = CE(S_F1, gts) + IOU(F.sigmoid(S_F1), gts)
            loss1b = CE(S_Fb, background) + IOU(S_Fb, background)
            loss2b = CE(S_F1b, background) + IOU(S_F1b, background)
            loss_gt = loss1 + loss2 + loss1b + loss2b

            loss_kd_logit = KLD_Loss(S_F / 2, T_F / 2, gts, 4) + \
                            KLD_Loss(S_F1 / 2, T_F1 / 2, gts, 4)
            loss_kd_logit_b = KLD_Loss(S_Fb / 2, T_Fb / 2, gts, 4) + \
                              KLD_Loss(S_F1b / 2, T_F1b / 2, gts, 4)

            loss_kd_decoder = MPSD_Loss1(S_f1, T_f1, S_a1, T_a1) + \
                              MPSD_Loss2(S_f2, T_f2, S_a2, T_a2) + \
                              MPSD_Loss3(S_f3, T_f3, S_a3, T_a3) + \
                              MPSD_Loss4(S_f4, T_f4, S_a4, T_a4)

            loss_cd = CD_Loss(S_fusion1, S_fusion2, S_fusion3, T_fusion1, T_fusion2, T_fusion3)
            loss_sal = loss_gt * 4 + loss_kd_logit * 4 + loss_kd_logit_b * 4 + loss_kd_decoder * 4 + loss_cd * 4  #0.8763

            # loss1 = CE(S_F, gts) + IOU(F.sigmoid(S_F), gts)
            # loss2 = CE(S_F1, gts) + IOU(F.sigmoid(S_F1), gts)
            # loss1b = CE(S_Fb, background) + IOU(S_Fb, background)
            # loss2b = CE(S_F1b, background) + IOU(S_F1b, background)
            # loss_gt = loss1 + loss2 + loss1b + loss2b
            #
            # loss_kd_logit = KLD_Loss(S_F/1.6, T_F/1.6, gts, 4) + \
            #                 KLD_Loss(S_F1/1.6, T_F1/1.6, gts, 4)
            # loss_kd_logit_b = KLD_Loss(S_Fb, T_Fb, gts, 4) + \
            #                   KLD_Loss(S_F1b, T_F1b, gts, 4)
            #
            # loss_kd_decoder = MPSD_Loss1(S_f1/1.2, T_f1/1.2, S_a1/1.2, T_a1/1.2) + \
            #                   MPSD_Loss2(S_f2/1.2, T_f2/1.2, S_a2/1.2, T_a2/1.2) + \
            #                   MPSD_Loss3(S_f3/1.2, T_f3/1.2, S_a3/1.2, T_a3/1.2) + \
            #                   MPSD_Loss4(S_f4/1.2, T_f4/1.2, S_a4/1.2, T_a4/1.2)
            #
            # loss_cd = CD_Loss(S_fusion1, S_fusion2, S_fusion3, T_fusion1, T_fusion2, T_fusion3)
            # loss_sal = loss_gt * 4 + loss_kd_logit * 4 + loss_kd_logit_b * 4 + loss_kd_decoder * 4 + loss_cd * 4

        elif 30 < epoch <= 60:
            loss1 = CE(S_F, gts) + IOU(F.sigmoid(S_F), gts)
            loss2 = CE(S_F1, gts) + IOU(F.sigmoid(S_F1), gts)
            loss1b = CE(S_Fb, background) + IOU(S_Fb, background)
            loss2b = CE(S_F1b, background) + IOU(S_F1b, background)
            loss_gt = loss1 + loss2 + loss1b + loss2b

            loss_kd_logit = KLD_Loss(S_F, T_F, gts, 4) + \
                            KLD_Loss(S_F1, T_F1, gts, 4)
            loss_kd_logit_b = KLD_Loss(S_Fb, T_Fb, gts, 4) + \
                              KLD_Loss(S_F1b, T_F1b, gts, 4)

            loss_kd_decoder = MPSD_Loss1(S_f1 / 2, T_f1 / 2, S_a1 / 2, T_a1 / 2) + \
                              MPSD_Loss2(S_f2 / 2, T_f2 / 2, S_a2 / 2, T_a2 / 2) + \
                              MPSD_Loss3(S_f3 / 2, T_f3 / 2, S_a3 / 2, T_a3 / 2) + \
                              MPSD_Loss4(S_f4 / 2, T_f4 / 2, S_a4 / 2, T_a4 / 2)

            loss_cd = CD_Loss(S_fusion1 / 2, S_fusion2 / 2, S_fusion3 / 2, T_fusion1 / 2, T_fusion2 / 2, T_fusion3 / 2)
            loss_sal = loss_gt * 4 + loss_kd_logit * 4 + loss_kd_logit_b * 4 + loss_kd_decoder * 4 + loss_cd * 4
        elif 60 < epoch <= 90:
            loss1 = CE(S_F, gts) + IOU(F.sigmoid(S_F), gts)
            loss2 = CE(S_F1, gts) + IOU(F.sigmoid(S_F1), gts)
            loss1b = CE(S_Fb, background) + IOU(S_Fb, background)
            loss2b = CE(S_F1b, background) + IOU(S_F1b, background)
            loss_gt = loss1 + loss2 + loss1b + loss2b

            loss_kd_logit = KLD_Loss(S_F, T_F, gts, 4) + \
                            KLD_Loss(S_F1, T_F1, gts, 4)
            loss_kd_logit_b = KLD_Loss(S_Fb, T_Fb, gts, 4) + \
                              KLD_Loss(S_F1b, T_F1b, gts, 4)

            loss_kd_decoder = MPSD_Loss1(S_f1, T_f1, S_a1 / 2, T_a1 / 2) + \
                              MPSD_Loss2(S_f2, T_f2, S_a2 / 2, T_a2 / 2) + \
                              MPSD_Loss3(S_f3, T_f3, S_a3 / 2, T_a3 / 2) + \
                              MPSD_Loss4(S_f4, T_f4, S_a4 / 2, T_a4 / 2)

            loss_cd = CD_Loss(S_fusion1, S_fusion2, S_fusion3, T_fusion1, T_fusion2, T_fusion3)
            loss_sal = loss_gt * 4 + loss_kd_logit * 4 + loss_kd_logit_b * 4 + loss_kd_decoder * 4 + loss_cd * 4

        else:
            loss1 = CE(S_F, gts) + IOU(F.sigmoid(S_F), gts)
            loss2 = CE(S_F1, gts) + IOU(F.sigmoid(S_F1), gts)
            loss1b = CE(S_Fb, background) + IOU(S_Fb, background)
            loss2b = CE(S_F1b, background) + IOU(S_F1b, background)
            loss_gt = loss1 + loss2 + loss1b + loss2b

            loss_kd_logit = KLD_Loss(S_F, T_F, gts, 4) + \
                            KLD_Loss(S_F1, T_F1, gts, 4)
            loss_kd_logit_b = KLD_Loss(S_Fb, T_Fb, gts, 4) + \
                              KLD_Loss(S_F1b, T_F1b, gts, 4)

            loss_kd_decoder = MPSD_Loss1(S_f1, T_f1, S_a1, T_a1) + \
                              MPSD_Loss2(S_f2, T_f2, S_a2, T_a2) + \
                              MPSD_Loss3(S_f3, T_f3, S_a3, T_a3) + \
                              MPSD_Loss4(S_f4, T_f4, S_a4, T_a4)

            loss_cd = CD_Loss(S_fusion1, S_fusion2, S_fusion3, T_fusion1, T_fusion2, T_fusion3)
            loss_sal = loss_gt * 4 + loss_kd_logit * 4 + loss_kd_logit_b * 4 + loss_kd_decoder * 4 + loss_cd * 4

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
