import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data1 = pd.read_csv("model_student") #with
data2 = pd.read_csv("model_student_wo")

s = list(data1.parameter)

# 绘制
label = list(data1.model)
# scatter = plt.scatter(x=list(data.parameter),y=list(data.accuracy),s=s,label ='a',c=color)
lin1, = plt.plot(list(data1.model), list(data1.mae),
         color = '#66B2FF',
         linestyle = '-',
         linewidth = 2,
         marker = 'p',
         markersize = 10,
         markeredgecolor = '#66B2FF',
         markerfacecolor = '#66B2FF')
lin2, = plt.plot(list(data2.model), list(data2.mae),
         color = '#CCFF99',
         linestyle = '-',
         linewidth = 2,
         marker = 'p',
         markersize = 10,
         markeredgecolor = '#CCFF99',
         markerfacecolor = '#CCFF99')

plt.legend(handles=[lin1, lin2], labels=['distill','fully'], loc='lower right')

plt.xlabel('BACKBONE', fontdict={'weight': 'normal', 'size': 13})#改变坐标轴标题字体
plt.ylabel('MAE', fontdict={'weight': 'normal', 'size': 13})#改变坐标轴标题字体
plt.show()

print("data.model", data1.model)

plt.show()
