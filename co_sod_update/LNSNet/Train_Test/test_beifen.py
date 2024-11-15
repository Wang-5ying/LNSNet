import os
import argparse

from tqdm import tqdm
import torch
from torch import nn

from dataset import get_loader
# from codes.GCoNet_plus.models.GCoNet import GCoNet
# from codes.bayibest82segformerbest.best.distill.teacher.ablation.backbone_pred_fusion.backbone_pred_fusion import M
from codes.bayibest82segformerbest.best.distill.student.student3_doconv_mulsup import S as M
from util import save_tensor_img
from config import Config
import torch.nn.functional as F


def main(args):
    # Init model
    config = Config()

    device = torch.device("cuda")
    # model = GCoNet()
    model = M()
    model = model.to(device)
    print('Testing with model {}'.format(args.ckpt))

    base_weights = torch.load(args.ckpt)
    # new_state_dict = OrderedDict()
    # for k, v in base_weights.items():
    #     name = k  # remove 'module.'
    #     new_state_dict[name] = v
    model.load_state_dict(base_weights)

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

        test_loader = get_loader(
            test_img_path, test_depth_path, test_gt_path, test_gt_path, test_gt_path, test_gt_path, test_gt_path, args.size, 1, istrain=False,
            shuffle=False, num_workers=8,
            pin=True)

        for batch in tqdm(test_loader):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            softgts = batch[2].to(device).squeeze(0)
            softgt2s = batch[3].to(device).squeeze(0)
            softgt3s = batch[4].to(device).squeeze(0)
            depths = batch[5].to(device).squeeze(0)
            edges = batch[6].to(device).squeeze(0)
            subpaths = batch[7]
            ori_sizes = batch[-1]
            with torch.no_grad():
                scaled_preds = model(inputs, depths, 1)[-1]  # 取得模型的最后一个输出作为预测结果
            print("scaled_preds", scaled_preds.size())
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
                    # res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                    #                                 align_corners=True)

                save_tensor_img(res, os.path.join(saved_root, subpath))


if __name__ == '__main__':
    # Parameter from command line
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',
                        default='GCoNet_plus',
                        type=str,
                        help="Options: '', ''")
    parser.add_argument('--testsets',
                        default='RGBD_CoSeg183',
                        # +CoSOD3k+Cosal2015 RGBD_CoSal1k' RGBD_CoSal150 RGBD_CoSeg183 DUTS_class
                        type=str,
                        help="Options: 'CoCA','Cosal2015','CoSOD3k','iCoseg','MSRC'")
    parser.add_argument('--size',
                        default=256,
                        type=int,
                        help='input size')
    parser.add_argument('--ckpt', default='/media/wby/shuju/backbone_cam/best/student/best_ep242_Smeasure0.8481.pth', type=str,
                        help='model folder')
    parser.add_argument('--pred_dir',
                        default='/media/wby/shuju/ckpt/M/RGBD_CoSeg183',
                        type=str, help='Output folder')

    args = parser.parse_args()

    main(args)
# [cost:20.5195s] RGBD_CoSeg183 (M-epRGBD_CoSeg183): 0.9118 max-Emeasure || 0.8532 S-measure  || 0.7761 max-fm || 0.0329 mae || 0.8780 mean-Emeasure || 0.7316 mean-fm.
