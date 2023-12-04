import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# labels = ['G1', 'G2', 'G3', 'G4', 'G5','G6'] # 级别
labels = ['S2MA', 'EDR', 'BBS', 'HAI', 'EMI', 'DAC', 'MC', 'DRER', 'LENO', 'CLA', 'HRT', 'ICON', 'PENet\n-Student', 'PENet\n-KD', 'PENet\n-Teacher']
# men_means = np.random.randint(20,35,size = 6)
Smeasure = [0.775, 0.811, 0.828, 0.718, 0.800, 0.824, 0.822, 0.844, 0.817, 0.835, 0.788, 0.843, 0.846, 0.856, 0.859]
# women_means = np.random.randint(20,35,size = 6)
maxE = [0.864, 0.893, 0.909, 0.829, 0.876, 0.911, 0.915, 0.933, 0.900, 0.920, 0.873, 0.925, 0.930, 0.938, 0.940]
maxF = [0.817, 0.850, 0.867, 0.803, 0.850, 0.875, 0.865, 0.891, 0.857, 0.878, 0.824, 0.884, 0.889, 0.902, 0.905]
x = np.arange(len(Smeasure))

plt.figure(figsize=(9,6))

width = 0.2
rects1 = plt.bar(x - width/3, Smeasure, width /3) # 返回绘图区域对象
rects2 = plt.bar(width/3, maxE, (width * 2 )/3)
rects3 = plt.bar((width * 2 )/3, maxE, width)

# 设置标签标题，图例
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(x,labels)
plt.legend(['Men','Women'])

# 添加注释
def set_label(rects):
    for rect in rects:
        height = rect.get_height() # 获取⾼度
        plt.text(x = rect.get_x() + rect.get_width()/2, # ⽔平坐标
                 y = height + 0.5, # 竖直坐标
                 s = height, # ⽂本
                 ha = 'center') # ⽔平居中

set_label(rects1)
set_label(rects2)
set_label(rects3)
plt.tight_layout() # 设置紧凑布局
plt.savefig('./分组带标签柱状图.png')
