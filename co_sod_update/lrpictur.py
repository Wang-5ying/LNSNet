import matplotlib.pyplot as plt
import json

import numpy as np

loss_values = []

with open('/storage/student2/wby/Project/HIVIT/mae/mae/output_dir/log0816.txt', 'r') as f:
  for line in f:
    data = json.loads(line)
    loss_values.append(data["train_loss"])

print(loss_values)

# 训练过程中的loss值，假设已经存储在列表'losses'中
epochs = range(1, len(loss_values) + 1)

# 使用Matplotlib绘制loss曲线
plt.plot(epochs, loss_values)
plt.grid(True)
# plt.yticks(np.arange(min(loss_values) + 0.5, max(loss_values) + 0.5, 0.5))
# 设置图表标题
plt.title('Training Loss Curve')

# 设置横轴和纵轴标签
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 显示图表
plt.show()