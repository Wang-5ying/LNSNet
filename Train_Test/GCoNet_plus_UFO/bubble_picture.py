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
c = np.random.choice(np.arange(4), 10)
# print(s)
# 气泡颜色
# color = np.random.rand(len(list(data.model)))
color = [0, 10, 20, 30, 40, 50, 60, 70, 80]
# print("color number", len(list(data.model)))
# print(color)
# 绘制
label = list(data.model)
print(label)
scatter = plt.scatter(x=list(data.parameter), y=list(data.accuracy), label=list(data.model), s=s, c=color)
# plt.legend(dem,labels = data.model)
# print("data.model", data.model)

plt.legend(labels=label, handles=scatter.legend_elements()[0], loc=4)
plt.xlabel('Size(MB)', fontdict={'weight': 'normal', 'size': 13})  # 改变坐标轴标题字体
plt.ylabel('maxF', fontdict={'weight': 'normal', 'size': 13})  # 改变坐标轴标题字体
# 显示
plt.grid()
plt.show()
