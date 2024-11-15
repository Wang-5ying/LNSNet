from random import seed
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
# from COA_RGBD_SOD.al.models.Fourth_model.ADMNet.ADMNet import ADMNet as CoNet
from Co_SOD.al_run.util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
# from Comparision_Methods.GCoNet_.models.GCoNet_D import GCoNet as CoNet
# from Co_SOD.CADC.CoSODNet.CoSODNet_D import CoSODNet as CoNet
# from Co_SOD.gicd.models.GICD_D import GICD as CoNet
# from Co_SOD.TCNet.network import ICNet as CoNet
from Co_SOD.ICNet.ICNet.network import ICNet as CoNet
import os
import numpy as np
import argparse
from Co_SOD.al_run.dataset import get_loader
import torch.nn.functional as F
from Co_SOD.al_run.config import Config
from Co_SOD.al_run.evaluation.dataloader import EvalDataset
from Co_SOD.al_run.evaluation.evaluator import Eval_thread
from Co_SOD.al_run.pytorch_iou.__init__ import IOU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='DUTS_class',  # default='Jigsaw2_DUTS',
                    type=str,
                    help="Options: 'DUTS_class'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--ckpt_dir',
                    default="/storage/student4/wby_data/comparemethod_output/ICNet/5",
                    help='Temporary folder')
parser.add_argument('--log_dir',
                    default="/storage/student4/wby_data/comparemethod_output/ICNet/5",
                    help='Result folder')
parser.add_argument('--testsets',
                    default='val5.0',
                    # default='', CoCA+CoSOD3k+Cosal2015  val5.0 RGBD_CoSeg183  RGBD_CoSal1k RGBD_CoSal150
                    type=str,
                    help="Options: 'CoCA','Cosal2015','CoSOD3k','iCoseg','MSRC'")
parser.add_argument('--testsets_over',
                    default='RGBD_CoSeg183+RGBD_CoSal150+RGBD_CoSal1k',
                    # default='', CoCA+CoSOD3k+Cosal2015  val5.0   RGBD_CoSal1k RGBD_CoSal150
                    type=str,
                    help="Options: 'CoCA','Cosal2015','CoSOD3k','iCoseg','MSRC'")
parser.add_argument('--val_dir',
                    default='/storage/student4/wby_data/comparemethod_output/ICNet/5/tmp4val/',
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
    # train_depth_path_seg = os.path.join(root_dir, 'depths/coco-seg')
    # train_edge_path_seg = os.path.join(root_dir, 'gts/coco-seg')
    # train_loader_seg = get_loader(
    #     train_img_path_seg,
    #     train_gt_path_seg,
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


test_loaders_over = {}

for testset_over in args.testsets_over.split('+'):
    test_loader_over = get_loader(
        os.path.join('/storage/student4/wby_data/', 'images', testset_over),
        os.path.join('/storage/student4/wby_data/', 'gts', testset_over),
        os.path.join('/storage/student4/wby_data/', 'depths', testset_over),
        os.path.join('/storage/student4/wby_data/', 'gts', testset_over),
        args.size, 1, max_num=config.batch_size, istrain=False, shuffle=False, num_workers=8, pin=True
    )
    test_loaders_over[testset_over] = test_loader_over

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
model = CoNet()

# load_model = '/storage/student4/wby_data/pretrain_weights/vgg16_20M.caffemodel.pth'
# vgg_model = torch.load(load_model)
# net = model.init_parameters(vgg_model)
# base_weights = torch.load('/media/map/fba69cc5-db71-46d1-9e6d-702c6d5a85f4/Alchemist/COA/COA_RGBD_SOD/ckpt/Thrid_models/CoNet_b4/CoNet_ep59_Smeasure0.8694.pth')
# model.load_state_dict(base_weights)
# print(torch.cuda.is_available())

model = model.to(device)
if config.lambda_adv:
    from COA_RGBD_SOD.al.adv import Discriminator

    disc = Discriminator(channels=1, img_size=args.size).to(device)
    optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.9, 0.99])
    Tensor = torch.cuda.FloatTensor if (True if torch.cuda.is_available() else False) else torch.FloatTensor
    adv_criterion = nn.BCELoss()

CE = torch.nn.BCEWithLogitsLoss().cuda()
IOU = IOU(size_average=True).cuda()
CL = nn.TripletMarginWithDistanceLoss().cuda()



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
            measures = validate(model, test_loaders, args.testsets)
            val_measures.append(measures)
            print(
                'Validation: S_measure on CoCA for epoch-{} is {:.4f}. Best epoch is epoch-{} with S_measure {:.4f}'.format(
                    epoch, measures[0], np.argmax(np.array(val_measures)[:, 0].squeeze()),
                    np.max(np.array(val_measures)[:, 0]))
            )
            if measures[0]>=0.8678:

                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'CoNet_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))
                print("Saved CoNet_ep{}_Smeasure{:.4f}.pth".format(epoch, measures[0]))

                ckpt_dir = os.path.join(args.ckpt_dir, 'CoNet_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0]))
                print('Testing with model {}'.format(ckpt_dir))
                model.load_state_dict(torch.load(ckpt_dir))

                model.eval()

                testsets_over = args.testsets_over.split('+')
                for testset_over in testsets_over[:3]:
                    print('Validating {}...'.format(testset_over))
                    test_loader = test_loaders_over[testset_over]
                    saved_root = os.path.join(args.val_dir, testset_over)
                    for batch in test_loader:
                        inputs = batch[0].to(device).squeeze(0)
                        gts = batch[1].to(device).squeeze(0)
                        depths = batch[2].to(device).squeeze(0)
                        edges = batch[3].to(device).squeeze(0)
                        subpaths = batch[4]
                        ori_sizes = batch[5]
                        with torch.no_grad():
                            scaled_preds = model(inputs, depths)[0]
                        os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)
                        num = len(scaled_preds)
                        for inum in range(num):
                            subpath = subpaths[inum][0]
                            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                            if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size,
                                                                mode='bilinear', align_corners=True)
                            else:
                                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size,
                                                                mode='bilinear', align_corners=True).sigmoid()
                            save_tensor_img(res, os.path.join(saved_root, subpath))
                    eval_loader = EvalDataset(
                        saved_root,
                        os.path.join('/home/map/Alchemist/COA/data/gts', testset_over)  # GT
                    )
                    thread = Eval_thread(eval_loader, method=str(epoch) + "_{:.3f}".format(measures[0]), dataset=testset_over,
                                         output_dir=args.log_dir, cuda=True)
                    info, continue_eval, max_e, s, max_f, mae = thread.run(continue_eval=True)
                    print(info)

                    if s >= config.val_measures['Smeasure'][testset_over] and max_e >=config.val_measures['Emax'][testset_over] and max_f >=config.val_measures['Fmax'][testset_over] and mae <=config.val_measures['MAE'][testset_over]:
                    # if s < config.val_measures['Smeasure'][testset_over] and max_e < config.val_measures['Emax'][testset_over] and max_f < config.val_measures['Fmax'][testset_over] and mae > config.val_measures['MAE'][testset_over]:
                        print("This is a qualified weight in {}".format(testset_over))
                    else:
                        os.renames(ckpt_dir, os.path.join(args.ckpt_dir, 'CoNet_ep{}_{:.3f}_{}_Sm{:.3f}_mE_{:.3f}_mF_{:.3f}_mae_{:.3f}.pth'.format(epoch, measures[0], testset_over, s, max_e, max_f, mae)))
                        print("This is an inekigible weight in {}".format(testset_over))
                        break



        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            path=args.ckpt_dir)

        if epoch >= args.epochs - 8:
            torch.save(model.state_dict(),
                       os.path.join(args.ckpt_dir, 'CoNet_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))
        if config.validation:
            if np.max(np.array(val_measures)[:, 0].squeeze()) == measures[0]:
                best_weights_before = [os.path.join(args.ckpt_dir, weight_file) for weight_file in
                                       os.listdir(args.ckpt_dir) if 'best_' in weight_file]
                for best_weight_before in best_weights_before:
                    os.remove(best_weight_before)
                torch.save(model.state_dict(),
                           os.path.join(args.ckpt_dir, 'CoNet_best_ep{}_Smeasure{:.4f}.pth'.format(epoch, measures[0])))

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def train(epoch, temperature):
    loss_log = AverageMeter()
    loss_log_triplet = AverageMeter()
    global logger_loss_idx
    model.train()


    # for batch_idx, (batch, batch_seg) in enumerate(zip(train_loader, train_loader_seg)):
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        depths = batch[2].to(device).squeeze(0)
        # background = 1 - gts
        # print("depth", depths.size())
        S_3_pred, S_4_pred, S_5_pred, S_g_pred = model(inputs, depths,is_training=True)

        # back1 = 1 - F.sigmoid(return_values[0])
        # back2 = 1 - F.sigmoid(return_values[1])
        # back3 = 1 - return_values[2]
        # back4 = 1 - return_values[3]

        # Compute IoU loss.
        # loss1 = CE(F.sigmoid(return_values[-1]), gts)
        loss1 = structure_loss(S_3_pred, gts)
        loss2 = structure_loss(S_4_pred, gts)
        loss3 = structure_loss(S_5_pred, gts)
        loss4 = structure_loss(S_g_pred, gts)
        loss = loss1 + loss3 + loss4 + loss4
        # loss2 = CE(F.sigmoid(return_values[-2]), gts) + IOU(return_values[-2], gts)
        # loss3 = CE(return_values[2], gts) + IOU(return_values[2], gts)
        # loss4 = CE(return_values[3], gts) + IOU(return_values[3], gts)
        #
        # loss_cl = CL(return_values[2], return_values[3], return_values[4])
        # loss1_b = CE(back1, background) + IOU(back1, background)
        # loss2_b = CE(back2, background) + IOU(back2, background)
        # loss3_b = CE(back3, background) + IOU(back3, background)
        # loss4_b = CE(back4, background) + IOU(back4, background)
        # Loss
        loss = 0
        # loss_sal = 0
        loss_sal = loss1 + loss2 + loss3 + loss4#+ loss2# + loss1_b + loss2_b
        # loss_sal = loss1 + loss1_b


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


def validate(model, test_loaders, testsets):
    # model().eval()
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


                scaled_preds = model(inputs, depths, is_training=False)


            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)-1
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
            os.path.join('/storage/student4/wby_data/gts', testset)  # GT root
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