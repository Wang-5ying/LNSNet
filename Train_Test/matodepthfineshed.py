import cv2
import scipy.io as scio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# 数据矩阵转图片的函数
def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


# 添加路径，metal文件夹下存放mental类的特征的多个.mat文件
datafolder = r'D:/新建文件夹/RINDNet-main/run/rindnet/depth/mat'
savefolder = r'D:/新建文件夹/RINDNet-main/run/rindnet/depth/out/'
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
    array_struct = scio.loadmat(each_mat)
    # print(array_struct)
    # 校验步骤
    array_data = array_struct['result']  # 取出需要的数字矩阵部分
    # print(array_data)
    # 校验步骤
    new_im = MatrixToImage(array_data)  # 调用函数
    plt.imshow(array_data, cmap=plt.cm.gray, interpolation='nearest')
    # new_im.show()
    # print(first_name)
    new_im.save(savefolder + first_name + '.jpg')  # 保存图片