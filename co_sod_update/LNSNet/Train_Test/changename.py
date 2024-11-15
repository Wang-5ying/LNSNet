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
path = '/home/wby/RGBT-EvaluationTools (copy)/SalMap/MLCP/RGBD_CoSal150'         #需要处理的文件路径
filelist = os.listdir(path) #打印所有文件夹下的内容，可以不要这3行代码

for file in filelist: #遍历所有文件
    Olddir=os.path.join(path,file) #原来的文件路径
    if os.path.isdir(Olddir): #如果是文件夹则跳过
        continue
    filename=os.path.splitext(file)[0] #分离文件名与扩展名;得到文件名
    filetype=os.path.splitext(file)[1] #文件扩展名
    Newdir=os.path.join(path,filename[:-14]+filetype) #filename[:-3]是原文件去掉倒数3位
    os.rename(Olddir,Newdir)#重命名，替换原图片





