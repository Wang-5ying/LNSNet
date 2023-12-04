import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("model_2023_第二篇")
# delete the data of United States
# data = data[data.state != 'United States']
# print(data.head(5))
# data = data[data.state != 'District of Columbia']
# print(data)
# 气泡值大小
s = list(data.parameter * 4)
# c = np.random.choice(np.arange(4), 16)
# print(s)
# 气泡颜色
# color = np.random.rand(len(list(data.model)))
color = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
# color = np.arange(13)
# print("color number", len(list(data.model)))
print(color)
# 绘制
label = list(data.model)
print(label)
scatter = plt.scatter(x=list(data.parameter), y=list(data.accuracy), label=list(data.model), marker='.', s=s, c=color)
# plt.legend(dem,labels = data.model)
print("scatter", scatter)
print("data.model", *scatter.legend_elements())

plt.legend(handles=scatter.legend_elements(num=color)[0], labels=label, loc=4)

plt.xlabel('Params(M)', fontdict={'weight': 'normal', 'size': 13})  # 改变坐标轴标题字体
plt.ylabel('S-measure', fontdict={'weight': 'normal', 'size': 13})  # 改变坐标轴标题字体
# 显示
plt.grid()
plt.show()
