from random import seed
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
from thop import profile
# from COA_RGBD_SOD.al.models.Segformer_GCN_teacher import RGBD_sal
from COA_RGBD_SOD.al.models.Segformerb5_GCN4_SA23 import RGBD_sal
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
from COA_RGBD_SOD.al.Loss.losses import KLDLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model
# from models GCoNet_plus import GCoNet_plus
# from COA_RGBD_SOD.al.models.GCoNet import GCoNet
# from codes.bayibest82segformerbest.best.newresdecoder4a614t4615622xiuz747117157261015cam11021108110911151116 import M
from COA_RGBD_SOD.al.pytorch_iou.__init__ import IOU

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
parser.add_argument('--ckpt_dir', default="/home/map/Alchemist/COA/COA_RGBD_SOD/ckpt/Selfkd/Segformerb5_GCN4_SA23/Pth_teacher",
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
# model_T =RGBD_sal()
model = RGBD_sal()
# model_T = RGBD_sal()
model.load_pre('/home/map/Alchemist/COA/COA_RGBD_SOD/al/Pretrain/segformer.b5.640x640.ade.160k.pth')

# base_weights = torch.load('/home/map/Alchemist/COA/COA_RGBD_SOD/ckpt/Segformer_GCN/Pth2/Segformer_GCN_best_ep146_Smeasure0.8668.pth')
# new_state_dict = OrderedDict()
# for k, v in base_weights.items():
#     name = k  # remove 'module.'
#     new_state_dict[name] = v
# model.load_state_dict()
# model_T.load_state_dict(new_state_dict)

# model.load_pre("/home/wby/segformer.b5.640x640.ade.160k.pth")  # 9.28好像忘记加载权重
# model.load_state_dict(torch.load("/media/wby/shuju/ckpt/best_ep15_Smeasure0.8391.pth"),strict=True)
# model.load_pre("/home/wby/Downloads/SegNext/mscan_l.pth")
# model_T = model_T.to(device)
model = model.to(device)
if config.lambda_adv:
    from COA_RGBD_SOD.al.adv import Discriminator

    disc = Discriminator(channels=1, img_size=args.size).to(device)
    optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.9, 0.99])
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()

CE = torch.nn.BCEWithLogitsLoss().cuda()
KLDLoss = KLDLoss().cuda()
IOU = IOU(size_average=True).cuda()
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
        if epoch >= args.epochs - 8:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'Segformer_GCN_ep{}.pth'.format(epoch)))

        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in
                                       os.listdir(args.ckpt_dir) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'Segformer_GCN_best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))


def train(epoch, temperature):
    loss_log = AverageMeter()
    loss_log_triplet = AverageMeter()
    global logger_loss_idx
    model.train()
    # model_T.eval()

    # for batch_idx, (batch, batch_seg) in enumerate(zip(train_loader, train_loader_seg)):
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        depths = batch[2].to(device).squeeze(0)
        edges = batch[3].to(device).squeeze(0)
        cls_gts = torch.LongTensor(batch[-1]).to(device)
        # print("gt--depth",gts.size(),edges.size())
        # print(cls_gts)

        # false_labels = model_T(inputs, depths)
        return_values = model(inputs, depths)
        # gts = false_labels[-1]
        # print(false_labels[0].shape)
        # print(return_values[0].shape)
        # print(gts.shape)
        # Loss
        cls_percentage = F.adaptive_avg_pool2d(gts, 4)
        loss = 0
        loss1 = CE(return_values[0], gts) + IOU(return_values[4], gts)
        loss2 = CE(return_values[1], gts) + IOU(return_values[5], gts)
        loss3 = CE(return_values[2], gts) + IOU(return_values[6], gts)
        loss4 = CE(return_values[3], gts) + IOU(return_values[7], gts)
        loss5 = KLDLoss(return_values[1], return_values[3], gts, 4)
        # loss5 = CE(return_values[8], gts) + IOU(return_values[12], gts)
        # print(return_values[3].size(), cls_percentage.size())
        # loss6 = CE(return_values[9], gts) + IOU(return_values[13], gts)
        # loss7 = CE(return_values[10], gts) + IOU(return_values[14], gts)
        # loss8 = CE(return_values[11], gts) + IOU(return_values[15], gts)
        # loss7 = CE(return_values[6], gts)
        # print("loss",return_values[3].size(),cls_gts.size())

        # print("loss-----", loss1,loss2,loss10*3,loss4)
        loss_sal = 8 * loss1 + 6 * loss2 + 4 * loss3 + 2 * loss4 + 10 * loss5
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
                scaled_preds = model(inputs, depths)[-1]
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
