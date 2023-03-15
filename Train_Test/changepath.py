import os
# import cv2
# import os
# import re
# path1 = '/media/wby/shuju/ckpt/M/RGBD_CoSal150/RGBD_CoSal150/airplane'
# pil1 = os.listdir(path1)
# pp = []
# for i in pil1:
#     a = path1 + '/' + i
#     print(i)
#     print(i[:11])
#     name = i[:11]
#     img = cv2.imread(a)
#     cv2.imwrite(i[:11]+"jpg", img)
# print(pp)
import os
path = '/media/wby/shuju/ckpt/M/RGBD_CoSal150'         #需要处理的文件路径 read
path2 = '/home/wby/COA-EvaluationTools/SalMap/CANet/RGBD_CoSal1k'

import os
import shutil

count = 0


def moveFiles(path, disdir):  # path为原始路径，disdir是移动的目标目录

    dirlist = os.listdir(path)
    for i in dirlist:
        child = os.path.join('%s/%s' % (path, i))
        if os.path.isfile(child):
            imagename, jpg = os.path.splitext(i)  # 分开文件名和后缀
            shutil.copy(child, os.path.join(disdir, imagename + ".png"))
            # 复制后改为原来图片名称
            # 也可以用shutil.move()
            continue
        moveFiles(child, disdir)


if __name__ == '__main__':
    rootimage = '/home/wby/CTNet-Results/CTNet-Results/RGBD_CoSeg183'  # 原始图片文件父目录   # CADC CADC+D CBCS CoEGNet EGNet_vgg GCoNet GCoNet+D GICD GICD+D HSCS ICNet ICNet+D ICS MCLP
    disdir = '/home/wby/COA-EvaluationTools/SalMap/CTNet/RGBD_CoSeg183'  # 移动到目标文件夹

    moveFiles(rootimage, disdir)






