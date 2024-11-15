import os
from PIL import Image, ImageEnhance
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
# from torchvision.transforms import InterpolationMode
import numbers
import random

from Co_SOD.al_run.preproc import cv_random_flip, random_crop, random_rotate, color_enhance, random_gaussian, random_pepper
from Co_SOD.al_run.config import Config
from scipy.io import loadmat

class CoData(data.Dataset):
    def __init__(self, image_root, label_root, depth_root, edge_root, image_size, max_num, is_train):
        class_list = os.listdir(image_root)  # 返回文件夹下所有文件名的列表（list）
        self.size_train = image_size
        self.size_test = image_size
        self.data_size = (self.size_train, self.size_train) if is_train else (self.size_test, self.size_test)
        self.image_dirs = list(map(lambda x: os.path.join(image_root, x), class_list))
        self.label_dirs = list(map(lambda x: os.path.join(label_root, x), class_list))
        self.depth_dirs = list(map(lambda x: os.path.join(depth_root, x), class_list))
        self.edge_dirs = list(map(lambda x: os.path.join(edge_root, x), class_list))
        self.max_num = max_num
        self.is_train = is_train
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_label = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ])
        self.transform_edge = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ])
        self.transform_depth = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
        ])
        self.load_all = False

    def __getitem__(self, item):

        names = os.listdir(self.image_dirs[item])

        num = len(names)

        image_paths = list(map(lambda x: os.path.join(self.image_dirs[item], x), names))

        label_paths = list(map(lambda x: os.path.join(self.label_dirs[item], x[:-4] + '.png'), names))
        depth_paths = list(map(lambda x: os.path.join(self.depth_dirs[item], x[:-4] + '.png'), names))
        edge_paths = list(map(lambda x: os.path.join(self.edge_dirs[item], x[:-4] + '.png'), names))
        if self.is_train:
            final_num = min(num, self.max_num)
            sampled_list = random.sample(range(num), final_num)
            new_image_paths = [image_paths[i] for i in sampled_list]
            new_label_paths = [label_paths[i] for i in sampled_list]
            new_depth_paths = [depth_paths[i] for i in sampled_list]
            new_edge_paths = [edge_paths[i] for i in sampled_list]
            image_paths = new_image_paths
            label_paths = new_label_paths
            depth_paths = new_depth_paths
            edge_paths = new_edge_paths

        else:
            final_num = num

        images = torch.Tensor(final_num, 3, self.data_size[1], self.data_size[0])
        depths = torch.Tensor(final_num, 3, self.data_size[1], self.data_size[0])
        labels = torch.Tensor(final_num, 1, self.data_size[1], self.data_size[0])
        edges = torch.Tensor(final_num, 1, self.data_size[1], self.data_size[0])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):
            if self.load_all:
                # TODO
                image = self.images_loaded[idx]
                label = self.labels_loaded[idx]
                depth = self.depths_loaded[idx]
                edge = self.edges_loaded[idx]
            else:
                if not os.path.exists(image_paths[idx]):
                    image_paths[idx] = image_paths[idx].replace('.jpg', '.png') if image_paths[idx][-4:] == '.jpg' else \
                    image_paths[idx].replace('.png', '.jpg')
                image = Image.open(image_paths[idx]).convert('RGB')
                if not os.path.exists(label_paths[idx]):
                    label_paths[idx] = label_paths[idx].replace('.jpg', '.png') if label_paths[idx][-4:] == '.jpg' else \
                    label_paths[idx].replace('.png', '.jpg')
                label = Image.open(label_paths[idx]).convert('L')
                if not os.path.exists(depth_paths[idx]):
                    depth_paths[idx] = depth_paths[idx].replace('.jpg', '.png') if image_paths[idx][-4:] == '.jpg' else \
                    depth_paths[idx].replace('.png', '.jpg')
                depth = Image.open(depth_paths[idx]).convert('RGB')
                if not os.path.exists(edge_paths[idx]):
                    edge_paths[idx] = edge_paths[idx].replace('.jpg', '.png') if edge_paths[idx][-4:] == '.jpg' else \
                    edge_paths[idx].replace('.png', '.jpg')
                edge = Image.open(edge_paths[idx]).convert('L')
            subpaths.append(
                os.path.join(image_paths[idx].split('/')[-2], image_paths[idx].split('/')[-1][:-4] + '.png'))
            ori_sizes.append((image.size[1], image.size[0]))

            # loading image and label
            if self.is_train:
                if 'flip' in Config().preproc_methods:
                    image, label, depth, edge = cv_random_flip(image, label, depth, edge)
                if 'crop' in Config().preproc_methods:
                    image, label, depth, edge = random_crop(image, label, depth, edge)
                if 'rotate' in Config().preproc_methods:
                    image, label, depth, edge = random_rotate(image, label, depth, edge)
                if 'enhance' in Config().preproc_methods:
                    image = color_enhance(image)
                if 'pepper' in Config().preproc_methods:
                    label = random_pepper(label)

            image, label, depth, edge = self.transform_image(image), self.transform_label(label), self.transform_depth(depth), self.transform_edge(edge)
            images[idx] = image
            labels[idx] = label
            depths[idx] = depth
            edges[idx] = edge

        if self.is_train:
            cls_ls = [item] * final_num
            return images, labels, depths, edges, subpaths, ori_sizes, cls_ls
        else:
            return images, labels, depths, edges, subpaths, ori_sizes

    def __len__(self):
        return len(self.image_dirs)


def get_loader(img_root, gt_root, depth_root, edge_root, img_size, batch_size, max_num=float('inf'), istrain=True, shuffle=False,
               num_workers=0, pin=False):
    dataset = CoData(img_root, gt_root, depth_root, edge_root, img_size, max_num, is_train=istrain)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader

device = torch.device("cuda")
if __name__ == '__main__':
    img_root = '/home/map/Alchemist/COA/data/images/DUTS_class'
    gt_root = '/home/map/Alchemist/COA/data/gts/DUTS_class'
    dep_root = '/home/map/Alchemist/COA/data/depths/DUTS_class'
    loader = get_loader(img_root, gt_root,dep_root, gt_root, 256, 1, 5)
    # for img, gt, subpaths, ori_sizes in loader:
    #     # print(img.size())
    #     # print(gt.size())
    #     print(subpaths)
    #     # print(ori_sizes)
    #     print(ori_sizes[0][0].item())
    #     break
    for batch_idx, batch in enumerate(loader):
        print(batch_idx)
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        depths = batch[2].to(device).squeeze(0)
        subpaths = batch[4]
        # print(img.size())
        # print(gt.size())
        print("subpaths len", len(subpaths))
        print(subpaths)
        # print(ori_sizes[0][0].item())
        # break