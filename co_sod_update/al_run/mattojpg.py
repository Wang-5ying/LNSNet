# #首先导入包
# from scipy.io import loadmat
# import cv2
# import scipy.io as scio
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import h5py
# # 数据矩阵转图片的函数
# def MatrixToImage(data):
#     data = data * 255
#     new_im = Image.fromarray(data.astype(np.uint8))
#     return new_im
# # 添加路径，文件夹下存放多个.mat文件
# savefolder = r'/home/wby/PycharmProjects/CoCA/data/depths/RGBD_CoSeg183/out/ball/'
# dataloader = '/home/wby/PycharmProjects/CoCA/data/depths/RGBD_CoSeg183/ball/RGBD_data_35.mat'
# #读取
#
# data=h5py.File(dataloader, 'r')
# x=list(data.keys())
# www=data[x[0]].value
# print(type(www))
# print(www.shape)
# # >>><class 'dict'>
# new_im = MatrixToImage(www)  # 调用函数
# plt.imshow(www, cmap=plt.cm.gray, interpolation='nearest')
# new_im.show()
# new_im.save(savefolder+"RGBD_data_35" + '.jpg')  # 保存图片
import cv2
import h5py
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# 数据矩阵转图片的函数
def MatrixToImage(data):
    max = data.max()
    data = data / max
    data = data * 255
    data = data.T
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# 添加路径，metal文件夹下存放mental类的特征的多个.mat文件
datafolder = '/media/wby/shuju/数据集/RGBD_CoSeg_Dataset/Depth_maps/white cap'
savefolder = '/home/wby/PycharmProjects/CoCA/data/depths/RGBD_CoSeg183/white cap/'
path = os.listdir(datafolder)
# print(os.path.splitext('100007.mat'))

for each_mat in path:
    # 先取一个文件做实验
    # each_mat='100007.mat'
    first_name, second_name = os.path.splitext(each_mat)
    # 拆分.mat文件的前后缀名字，
    each_mat = os.path.join(datafolder, each_mat)
    # print(each_mat)
    # 校验步骤，输出应该是文件名
    # array_struct = scio.loadmat(each_mat)
    array_struct=h5py.File(each_mat, 'r')
    # print(array_struct['depth'].shape)
    # 校验步骤
    # array_data = array_struct['result']  # 取出需要的数字矩阵部分
    x = list(array_struct.keys())
    # print(x[0])
    array_data=array_struct[x[0]].value

    # print(array_data)
    # 校验步骤
    new_im = MatrixToImage(array_data)  # 调用函数
    plt.imshow(array_data, cmap=plt.cm.gray, interpolation='nearest')
    # new_im.show()
    # print(first_name)
    new_im.save(savefolder + first_name + '.png')  # 保存图片