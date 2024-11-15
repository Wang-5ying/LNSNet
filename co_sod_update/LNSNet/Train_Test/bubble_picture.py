import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("model_2023")
# delete the data of United States
# data = data[data.state != 'United States']
# print(data.head(5))
# data = data[data.state != 'District of Columbia']
# print(data)
# 气泡值大小
s = list(data.parameter)
# print(s)
# 气泡颜色
# color = np.random.rand(len(list(data.model)))
color = [5, 10, 25,40,55,70,85,100,115]
# print("color number", len(list(data.model)))
# print(color)
# 绘制
label = list(data.model)
scatter = plt.scatter(x=list(data.parameter),y=list(data.accuracy),s=s,label ='a',c=color)
# plt.legend(dem,labels = data.model)
print("data.model", data.model)
plt.legend(handles = scatter.legend_elements()[0], labels = label)
plt.xlabel('Size(MB)', fontdict={'weight': 'normal', 'size': 13})#改变坐标轴标题字体
plt.ylabel('F-measure', fontdict={'weight': 'normal', 'size': 13})#改变坐标轴标题字体
# 显示
plt.grid()
plt.show()
