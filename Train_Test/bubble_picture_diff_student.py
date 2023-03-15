import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data1 = pd.read_csv("model_student") #with
data2 = pd.read_csv("model_student_wo")

# data3 = pd.read_csv("model_student2_w")
# data4 = pd.read_csv("model_student2")
#
# data5 = pd.read_csv("model_student3_w")
# data6 = pd.read_csv("model_student3")
# 气泡值大小
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

# plt.plot(list(data3.model), list(data3.mae),
#          color = '#99FF99',
#          linestyle = '-.',
#          linewidth = 2,
#          marker = 'p',
#          markersize = 8,
#          markeredgecolor = '#99FF99',
#          markerfacecolor = '#99FF99')
# plt.plot(list(data4.model), list(data4.mae),
#          color = '#99FF99',
#          linestyle = '-',
#          linewidth = 2,
#          marker = 'p',
#          markersize = 8,
#          markeredgecolor = '#99FF99',
#          markerfacecolor = '#99FF99')
#
# plt.plot(list(data5.model), list(data5.mae),
#          color = '#3399FF',
#          linestyle = '-.',
#          linewidth = 2,
#          marker = 'p',
#          markersize = 8,
#          markeredgecolor = '#3399FF',
#          markerfacecolor = '#3399FF')
# plt.plot(list(data6.model), list(data6.mae),
#          color = '#3399FF',
#          linestyle = '-',
#          linewidth = 2,
#          marker = 'p',
#          markersize = 8,
#          markeredgecolor = '#3399FF',
#          markerfacecolor = '#3399FF')
plt.legend(handles=[lin1, lin2], labels=['distill','fully'], loc='lower right')

plt.xlabel('BACKBONE', fontdict={'weight': 'normal', 'size': 13})#改变坐标轴标题字体
plt.ylabel('MAE', fontdict={'weight': 'normal', 'size': 13})#改变坐标轴标题字体
plt.show()

print("data.model", data1.model)



# 显示
# plt.grid()
plt.show()
