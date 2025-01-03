import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.optim as optim

from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
import argparse
# from dataset import get_loader
import torch.nn.functional as F
from config import Config
from loss import DSLoss
from Co_SOD.al_run.dataset import get_loader
from evaluation.evaluator import Eval_thread
from Co_SOD.al_run.evaluation.dataloader import EvalDataset



# model
# from models GCoNet_plus import GCoNet_plus
# teacher
# from codes.bayibest82segformerbest.best.distill.teacher.ablation.backbone_pred.backbone_pred import M as S
# student
# from codes.bayibest82segformerbest.best.distill.student.student3_doconv_mulsup_b0 import S
from Co_SOD.LNSNet.student.student2_b3 import M as S
from Co_SOD.LNSNet.teacher import M as teacher
# from codes.bayibest82segformerbest.best.distill.ablation.backbone import M as S
# from codes.pytorch_iou.__init__ import IOU
from Co_SOD.LNSNet.student.demo import KLDLoss#, OFD
# from 文献代码.KnowledgeDistillationZoomaster.kd_losses.ofd import OFD
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
parser.add_argument('--epochs', default=300, type=int)  # 320
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
parser.add_argument('--ckpt_dir', default="/storage/student4/wby_data/distill_ckpt6", help='Temporary folder')
parser.add_argument('--ckpt_dir2', default="/media/wby/shuju/ckpt2", help='Temporary folder')
parser.add_argument('--testsets',
                    default='val5.0',  # default='', CoCA+CoSOD3k+Cosal2015    val5.0
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
    root_dir = '/storage/student4/wby_data/'
    train_img_path = os.path.join(root_dir, 'images/DUTS_class')
    train_gt_path = os.path.join(root_dir, 'gts/DUTS_class')
    train_depth_path = os.path.join(root_dir, 'depths/DUTS_class')
    train_edge_path = os.path.join(root_dir, 'images/DUTS_class')
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
    # train_img_path_seg = os.path.join(root_dir, 'images/coco-seg')
    # train_gt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    # train_softgt_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    # train_softgt2_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    # train_softgt3_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    # train_depth_path_seg = os.path.join(root_dir, 'depths/coco-seg')
    # train_edge_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    # train_loader_seg = get_loader(
    #     train_img_path_seg,
    #     train_gt_path_seg,
    #     train_softgt_path_seg,
    #     train_softgt2_path_seg,
    #     train_softgt3_path_seg,
    #     train_depth_path_seg,
    #     train_edge_path_seg,
    #     args.size,
    #     1,
    #     max_num=config.batch_size,
    #     istrain=True,
    #     shuffle=True,
    #     num_workers=8,
    #     pin=True
    # )
elif args.trainset == 'CoCA':
    root_dir = '/storage/student4/wby_data/'
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
        os.path.join('/storage/student4/wby_data/', 'images', testset),
        os.path.join('/storage/student4/wby_data/', 'gts', testset),
        os.path.join('/storage/student4/wby_data/', 'depths', testset),
        os.path.join('/storage/student4/wby_data/', 'gts', testset),
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
# model =M()
model2 = S()
teachers = teacher()
# model2.load_pre("/home/wby/Downloads/convnextv2_base_22k_224_ema.pt")
# model2.load_pre("/home/wby/Downloads/cotnext101_2x48d.pth.tar","/home/wby/segformer.b5.640x640.ade.160k.pth")  # 9.28好像忘记加载权重
# model2.load_pre("/home/wby/Downloads/pvt_large_iter_40000.pth")
model2.load_pre("/storage/student4/wby_data/segformer_weights/segformer.b3.512x512.ade.160k.pth")
teachers.load_state_dict(torch.load("/storage/student4/wby_data/Fourth_model_Teacher_weights/best_ep49_Smeasure0.8732.pth"), strict=True)
# model2.load_state_dict(torch.load("/media/wby/shuju/ckpt/best_ep268_Smeasure0.8545.pth"),strict=True)
# model.load_pre("/home/wby/Downloads/SegNext/mscan_l.pth")
# model = model.to(device)
model2 = model2.to(device)
teachers = teachers.to(device)
if config.lambda_adv:
    from adv import Discriminator

    disc = Discriminator(channels=1, img_size=args.size).to(device)
    optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.9, 0.99])
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()

# teacher
CE = torch.nn.BCEWithLogitsLoss().cuda()
kld = KLDLoss()
# ofd = OFD(1,1)
# IOU = IOU(size_average=True).cuda()
# all_params = model.parameters()
# student
all_params2 = model2.parameters()
# Setting optimizer - teacher
optimizer = optim.Adam(params=all_params2, lr=config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_step_size, gamma=0.1)
# Setting optimizer - student
# optimizer2 = optim.Adam(params=all_params2, lr=config.lr)
# scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=config.decay_step_size, gamma=0.1)

# Why freeze the backbone?...
if config.freeze:
    for key, value in model2.named_parameters():
        if 'bb' in key and 'bb.conv5.conv5_3' not in key:
            value.requires_grad = False

# log model and optimizer params
logger.info("Model details:")
logger.info(model2)
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
    # val_measures2 = []
    global temperature
    temperature = 34
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            model2.load_state_dict(torch.load(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(epoch, temperature)
        print('temperature:', temperature)
        if temperature != 1:
            temperature -= 3
            print('Change temperature to:', str(temperature))
        if config.validation:
            # teacher
            measures = validate(model2, test_loaders, args.testsets, temperature)
            val_measures.append(measures)
            # student
            # measures2 = validate(model2, test_loaders, args.testsets, temperature)
            # val_measures2.append(measures2)
            print('teacher - Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with '
                  'S_measure {:.4f}'.format(
                epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                np.max(np.array(val_measures)[:, 0]))
            )
            # print('teacher - Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with '
            #       'S_measure {:.4f}'.format(
            #     epoch, measures[0], np.argmin(np.array(val_measures)[:, 0].squeeze()),
            #     np.mean(np.array(val_measures)[:, 0]))
            # )
            # print(
            #     'Student - Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with '
            #     'S_measure {:.4f}'.format(
            #         epoch, measures2[0], np.argmax(np.array(val_measures2)[:, 0].squeeze()),
            #         np.max(np.array(val_measures2)[:, 0]))
            # )
            # print(
            #     'Validation: MAE on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with MAE {:.4f}'.format(
            #         epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
            #         np.max(np.array(val_measures)[:, 0]))
            # )
        # Save checkpoint
        # teacher
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model2.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            path=args.ckpt_dir)

        # teacher
        if epoch > 20:
            torch.save(model2.state_dict(), os.path.join(args.ckpt_dir, 'ep{}.pth'.format(epoch)))
        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in
                                       os.listdir(args.ckpt_dir) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model2.state_dict(),
                           os.path.join(args.ckpt_dir, 'best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))
        # if config.validation:
        #     if np.min(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
        #         best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in
        #                                os.listdir(args.ckpt_dir) if 'best_' in weight_file]
        #         for best_weight_before in best_weights_before:
        #             os.remove(best_weight_before)
        #         torch.save(model2.state_dict(),
        #                    os.path.join(args.ckpt_dir, 'best_ep{}smeasure{:.4f}.pth'.format(epoch, measures[0])))


def train(epoch, temperature):
    loss_log = AverageMeter()
    loss_log_triplet = AverageMeter()
    global logger_loss_idx
    # teacher
    model2.train()
    # student
    # model2.train()

    # for batch_idx, (batch, batch_seg) in enumerate(zip(train_loader, train_loader_seg)):
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        depths = batch[2].to(device).squeeze(0)

        # teacher
        with torch.no_grad():
            teacher_rte, teacher_dte, decode, return_values_T = teachers(inputs, depths)

        return_values, r, d, xdecode, res1 = model2(inputs, depths)

        if epoch < 40:
            kl1 = kld(xdecode[0]/temperature, decode[0]/temperature, gts/temperature, 3)
            kl2 = kld(xdecode[1]/temperature, decode[1]/temperature, gts/temperature, 3)
            kl3 = kld(xdecode[2]/temperature, decode[2]/temperature, gts/temperature, 3)
            kl4 = kld(xdecode[3]/temperature, decode[3]/temperature, gts/temperature, 3)
            kl = (kl1 + kl2 + kl3 + kl4)*2
        else:
            kl1 = CE(return_values[0], gts)
            kl2 = CE(return_values[2], gts)
            kl3 = CE(return_values[4], gts)
            kl = (kl1 + kl2 + kl3)*2

        b3l = CE(res1, gts)

        cls_percentage = F.adaptive_avg_pool2d(gts, 32)

        loss = 0

        # 1208 plan c  ---- best for distill ----------------------------------------------------------------
        loss1_gt = CE(return_values[0], gts)  # 硬目标, 老师也有一定的错误率
        loss1_te = kld(return_values[1], F.interpolate(return_values_T[0], size=256)/4, gts, 3)  # 软目标

        loss2_gt = CE(return_values[2], gts)  # 硬目标
        loss2_te = kld(return_values[3], F.interpolate(return_values_T[1], size=256)/4, gts, 3)  # 软目标

        loss3_gt = CE(return_values[4], gts)  # 硬目标
        loss3_te = kld(return_values[5], F.interpolate(return_values_T[2], size=256)/4, gts, 3)  # 软目标

        loss4_gt = CE(return_values[6], cls_percentage)  # 硬目标
        loss4_te = kld(return_values[7], cls_percentage/4 ,cls_percentage, 3)  # 软目标
        #
        flag1 = False
        flag2 = False
        flag3 = False
        flag4 = False
        # 1
        # print(decode[0].size(), gts.size())
        teacher_loss = CE(return_values_T[0], gts)
        student_loss = CE(return_values[0], gts)
        if teacher_loss < student_loss + 0.2 and epoch < 40:
            loss_sal1 = loss1_gt * 0.0625 + loss1_te * (
                    1 - 0.0625)  # !!!!    # sub/4 loss1 * 0.0625 + loss2 * (1 - 0.0625)  # !!!!
            flag1 = True
        else:
            loss_sal1 = loss1_gt * 0.0625  # * 0.0625
        # 2
        teacher_loss = CE(return_values_T[1], gts)
        student_loss = CE(return_values[2], gts)
        if teacher_loss < student_loss + 0.2 and epoch < 40:
            loss_sal2 = loss2_gt * 0.0625 + loss2_te * (
                    1 - 0.0625)  # !!!!    # sub/4 loss1 * 0.0625 + loss2 * (1 - 0.0625)  # !!!!
            flag2 = True
        else:
            loss_sal2 = loss2_gt * 0.0625  # * 0.0625
        # 3
        teacher_loss = CE(return_values_T[2], gts)
        student_loss = CE(return_values[4], gts)
        if teacher_loss < student_loss + 0.2 and epoch < 40:
            loss_sal3 = loss3_gt * 0.0625 + loss3_te * (
                    1 - 0.0625)  # !!!!    # sub/4 loss1 * 0.0625 + loss2 * (1 - 0.0625)  # !!!!
            flag3 = True
        else:
            loss_sal3 = loss3_gt * 0.0625  # * 0.0625
        #4
        # teacher_loss = CE(cam_4, cls_percentage)
        # student_loss = CE(return_values[6], cls_percentage)
        # if teacher_loss < student_loss + 0.2 and epoch < 40:
        loss_sal4 = loss4_gt * 0.0625 + loss4_te * (
                    1 - 0.0625)  # !!!!    # sub/4 loss1 * 0.0625 + loss2 * (1 - 0.0625)  # !!!!
            # flag3 = True
        # else:
        #     loss_sal4 = loss4_gt * 0.0625  # * 0.0625




        # loss_sal = 30 * loss_sal1 + loss_sal2  # loss_sal1 + loss_sal2
        # print("1",flag1, loss_sal1, flag2, loss_sal2)
        # print("2",flag3, loss_sal3, flag4, loss_sal4)
        # print(kl*2,loss_sal4*10,(loss_sal1 + loss_sal2 + loss_sal3) *10)
        # print(b3l, kl*10, (loss_sal1 + loss_sal2 + loss_sal3) *10)
        loss_sal = (loss_sal1 + loss_sal2 + loss_sal3) *10 + kl * 10 + loss_sal4 + b3l
        # print((loss_sal1 + loss_sal2 + loss_sal3) *10, kl*2)
        # -----------------------------------------------------------------------------------------------------------------------------------------------
        # loss_sal = CE(return_values[0], gts) + CE(return_values[2], gts) + CE(return_values[4], gts) + CE(return_values[6], cls_percentage)
        loss = loss + loss_sal
        # teacher student

        # loss2 = loss_s

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
                # teacher
                scaled_preds = model(inputs, depths)[-1]  # 取得模型的最后一个输出作为预测结果，和test一样，所以需要在模型里要注意输出
                # # student
                # scaled_preds2 = model2(inputs,depths)
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
            os.path.join('/storage/student4/wby_data/gts', testset)  # GT
        )
        evaler = Eval_thread(eval_loader, cuda=True)
        # Use S_measure for validation
        s_measure = evaler.Eval_Smeasure()
        # mae = evaler.Eval_mae()
        # print("mae", mae)
        if s_measure > config.val_measures['Smeasure']['val_38'] and 0:  # CoCA
        # if mae > config.val_measures['Emeasure']['val_38'] and 0:
            print("llll")
            # TODO: evluate others measures if s_measure is very high.
            e_max = evaler.Eval_Emeasure().max().item()
            f_max = evaler.Eval_fmeasure().max().item()
            mae_max = evaler.Eval_mae().max().item()
            print('Emax: {:4.f}, Fmax: {:4.f}'.format(e_max, f_max))
        measures.append(s_measure)
        # measures.append(mae)
    model.train()
    return measures


if __name__ == '__main__':
    main()
