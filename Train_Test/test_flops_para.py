import os
import argparse
import time
from collections import OrderedDict

from tqdm import tqdm
import torch
from torch import nn

from dataset import get_loader
# from codes.GCoNet_plus.models.GCoNet import GCoNet
# from 文献代码.DenseDepth.PyTorch.model import PTModel
# from codes.bayibest82segformerbest.best.distill.student.student2 import M
# from 文献代码.深度估计.NeWCRFs.newcrfs.networks.NewCRFDepth import NewCRFDepth
# from codes.bayibest82segformerbest.best.distill.teacher.teacher import M
# from 对比文章.ICNet.ICNet.networkwD import ICNet
# from codes.GCoNet.models.GCoNet import GCoNet
from 对比文章.GICD.models.GICD import GICD
# from codes.bayibest82segformerbest.best.distill.teacher.ablation_定稿.ablation.backbone import M
# from codes.bayibest82segformerbest.best.student2 import S as M
# from codes.bayibest82segformerbest.best.distill.student.student3_doconv_mulsup import S as M
# from codes.bayibest82segformerbest.best.distill.teacher.ablation.backbone_cam.backbone_cam import M
# from 对比文章.EGNet.model import build_model
# from codes.GCoNet.models.GCoNet import GCoNet as M
# TEST FOR 对比文章
# from 对比文章.GICD.models.GICD import GICD as M
from util import save_tensor_img
from config import Config
import torch.nn.functional as F
from thop import profile


def main(args):
    # Init model
    config = Config()

    device = torch.device("cuda")
    # model = GCoNet()
    # model = M()
    model = GICD()
    # model = NewCRFDepth(version='large07', inv_depth=False, max_depth=10)
    # model = build_model(base_model_cfg='vgg')
    model = model.to(device)
    print('Testing with model {}'.format(args.ckpt))


    # new_state_dict = OrderedDict()
    # for k, v in base_weights.items():
    #     name = 'ginet.' + k  # remove 'module.'
    #     new_state_dict[name] = v
    # load pth
    # base_weights = torch.load(args.ckpt)
    # model.load_state_dict(base_weights)
    model.eval()

    for testset in args.testsets.split('+'):
        print('Testing {}...'.format(testset))
        root_dir = '/home/wby/PycharmProjects/CoCA/data/'
        if testset == 'CoCA':
            test_img_path = os.path.join(root_dir, 'images/CoCA')
            test_gt_path = os.path.join(root_dir, 'gts/CoCA')
            test_depth_path = os.path.join(root_dir, 'depths/CoCA')
            saved_root = os.path.join(args.pred_dir, 'CoCA')
        elif testset == 'CoSOD3k':
            test_img_path = os.path.join(root_dir, 'images/CoSOD3k')
            test_gt_path = os.path.join(root_dir, 'gts/CoSOD3k')
            saved_root = os.path.join(args.pred_dir, 'CoSOD3k')
        elif testset == 'Cosal2015':
            test_img_path = os.path.join(root_dir, 'images/Cosal2015')
            test_gt_path = os.path.join(root_dir, 'gts/Cosal2015')
            saved_root = os.path.join(args.pred_dir, 'Cosal2015')
        elif testset == 'RGBD_CoSal150':
            test_img_path = os.path.join(root_dir, 'images/RGBD_CoSal150')
            test_depth_path = os.path.join(root_dir, 'depths/RGBD_CoSal150')
            test_gt_path = os.path.join(root_dir, 'gts/RGBD_CoSal150')
            saved_root = os.path.join(args.pred_dir, 'RGBD_CoSal150')
        elif testset == 'RGBD_CoSal1k':
            test_img_path = os.path.join(root_dir, 'images/RGBD_CoSal1k')
            test_depth_path = os.path.join(root_dir, 'depths/RGBD_CoSal1k')
            test_gt_path = os.path.join(root_dir, 'gts/RGBD_CoSal1k')
            saved_root = os.path.join(args.pred_dir, 'RGBD_CoSal1k')
        elif testset == 'RGBD_CoSeg183':
            test_img_path = os.path.join(root_dir, 'images/RGBD_CoSeg183')
            test_depth_path = os.path.join(root_dir, 'depths/RGBD_CoSeg183')
            test_gt_path = os.path.join(root_dir, 'gts/RGBD_CoSeg183')
            saved_root = os.path.join(args.pred_dir, 'RGBD_CoSeg183')
        elif testset == 'DUTS_class':
            test_img_path = os.path.join(root_dir, 'images/DUTS_class')
            test_depth_path = os.path.join(root_dir, 'depths/DUTS_class')
            test_gt_path = os.path.join(root_dir, 'gts/DUTS_class')
            saved_root = os.path.join(args.pred_dir, 'DUTS_class')
        else:
            print('Unkonwn test dataset')
            print(args.dataset)
        #     img_root,       gt_root,       softgt_root, softgt2_root, softgt3_root, depth_root, edge_root,
        # 小的三个测试集合
        test_loader = get_loader(
            test_img_path, test_gt_path,  test_gt_path, test_gt_path, test_gt_path, test_depth_path, test_gt_path,  args.size, 1, istrain=False,
            shuffle=False, num_workers=8,
            pin=True)
        # 大的三个测试集合
        # test_loader = get_loader(
        #     test_img_path, test_gt_path, test_gt_path, test_gt_path, test_gt_path, test_gt_path, test_gt_path,  # ！没有depth这里用gt代替，下面的输入记得修改
        #     args.size, 1, istrain=False,
        #     shuffle=False, num_workers=8,
        #     pin=True)
        # test_loader = get_loader(
        #     test_img_path, test_depth_path, test_gt_path, test_gt_path, test_gt_path, test_gt_path, test_gt_path,
        #     args.size, 1, istrain=False,
        #     shuffle=False, num_workers=8,
        #     pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device)
            gts = batch[1].to(device).squeeze(0)
            softgts = batch[2].to(device).squeeze(0)
            softgt2s = batch[3].to(device).squeeze(0)
            softgt3s = batch[4].to(device).squeeze(0)
            depths = batch[5].to(device)
            edges = batch[6].to(device).squeeze(0)
            subpaths = batch[7]
            ori_sizes = batch[-1]
            # print(depths.size())

            # for GICD
            # B, C, H, W = inputs.size()
            inputs = inputs.transpose(0,1)
            depths = depths.transpose(0, 1)
            # B, C, H, W = depths.size()
            # depths = depths.resize(B, 1, C, H, W)
            print(inputs.size(), depths.size())
            with torch.no_grad():
                start = time.clock()
                out = model(inputs, depths)
                scaled_preds = out[-1][-1]  # 取得模型的最后一个输出作为预测结果
                print(scaled_preds.size())
            # for i in scaled_preds:
            #     print("scaled_preds", i.size())
            os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]), exist_ok=True)

            num = len(scaled_preds)
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                if config.db_output_refiner or (not config.refine and config.db_output_decoder):
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True)
                else:
                    ### 原本的输出到原图尺寸  GCoNet+在文中将预测图拉回到原图尺寸进行评估
                    # print("22") # this
                    res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                    align_corners=True).sigmoid()


                    # res = scaled_preds[inum].unsqueeze(0).sigmoid()
                    # print("res",res.size())
                    # test for cam
                    # res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                    #                                 align_corners=True)/4
                    # res = F.sigmoid(nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                    #                                 align_corners=True)/4)
                    # # res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                    #                                 align_corners=True)

                save_tensor_img(res, os.path.join(saved_root, subpath))
        a = torch.randn(1,1,3,256,256).cuda()
        b = torch.randn(1,1,1,256,256).cuda()
        rgb = torch.randn(inputs.size()).cuda()
        # dep = torch.randn(depths.size()).cuda()
        dep = torch.randn(depths.size()).cuda()
        dep2 = torch.randn(gts.size()).cuda()
        print(a.size(), a.size())
        flops, params = profile(model, (a, a))
        print('Flops', flops /1e9, 'G')
        print('Params: ', params / 1e6, 'M')


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet_plus',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--testsets',
                        default='RGBD_CoSal150',
                        # +CoSOD3k+Cosal2015 RGBD_CoSal150' RGBD_CoSal1k  RGBD_CoSal150 RGBD_CoSeg183 DUTS_class
                        type=str,
                        help="Options: 'CoCA','Cosal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size',
                        default=256,  # 256
                        type=int,
                        help='input size')
    parser.add_argument('--ckpt', default='/home/wby/Downloads/best_ep49_Smeasure0.8732.pth', type=str,  # 71-8627
                        help='model folder')
    parser.add_argument('--pred_dir',
                        default='/media/wby/shuju/ckpt/M/RGBD_CoSal150',
                        type=str, help='Output folder')

    args = parser.parse_args()

    main(args)
# [cost:20.5195s] RGBD_CoSeg183 (M-epRGBD_CoSeg183): 0.9118 max-Emeasure || 0.8532 S-measure  || 0.7761 max-fm || 0.0329 mae || 0.8780 mean-Emeasure || 0.7316 mean-fm.
