from scipy import stats
import numpy as np

# 成对样本t检验 MYmodel
sample1 = [0.8738, 0.8731, 0.8777, 0.8884] # our model
sample2 = [0.861, 0.864, 0.8666, 0.8641] # 对比模型 -TCNet
# sample2 = [0.8035, 0.781, 0.7842, 0.8051] # 对比模型 -CADC
# sample2 = [0.6606, 0.6857, 0.7074, 0.6502] # 对比模型 -ICNet
# sample2 = [0.6512, 0.5303, 0.3838, 0.3844] # 对比模型 -GICD

# # 成对样本t检验 TCNet
# sample1 = [0.861, 0.864, 0.8666, 0.8641]
# # sample2 = [0.8738, 0.8731, 0.8777, 0.8884] # our model
# # sample2 = [0.8035, 0.781, 0.7842, 0.8051] # 对比模型 -CADC
# # sample2 = [0.6606, 0.6857, 0.7074, 0.6502] # 对比模型 -ICNet
# sample2 = [0.6512, 0.5303, 0.3838, 0.3844] # 对比模型 -GICD

# # 成对样本t检验 CADC
# sample1 = [0.8035, 0.781, 0.7842, 0.8051] # 对比模型 -CADC
# # sample2 = [0.861, 0.864, 0.8666, 0.8641]  # TCNet
# # sample2 = [0.8738, 0.8731, 0.8777, 0.8884] # our model
# # sample2 = [0.6606, 0.6857, 0.7074, 0.6502] # 对比模型 -ICNet
# sample2 = [0.6512, 0.5303, 0.3838, 0.3844] # 对比模型 -GICD

# # 成对样本t检验 ICNet
# sample1 = [0.6606, 0.6857, 0.7074, 0.6502] # 对比模型 -ICNet
# # sample2 = [0.8738, 0.8731, 0.8777, 0.8884] # our model
# # sample2 = [0.861, 0.864, 0.8666, 0.8641] # 对比模型 -TCNet
# # sample2 = [0.8035, 0.781, 0.7842, 0.8051] # 对比模型 -CADC
# sample2 = [0.6512, 0.5303, 0.3838, 0.3844] # 对比模型 -GICD

# # # 成对样本t检验 GICD
# sample1 = [0.6512, 0.5303, 0.3838, 0.3844] # 对比模型 -GICD
# # sample2 = [0.8738, 0.8731, 0.8777, 0.8884] # our model
# # sample2 = [0.861, 0.864, 0.8666, 0.8641] # 对比模型 -TCNet
# # sample2 = [0.8035, 0.781, 0.7842, 0.8051] # 对比模型 -CADC
# sample2 = [0.6606, 0.6857, 0.7074, 0.6502] # 对比模型 -ICNet



from scipy.stats import t
import math
import numpy as np

if __name__ == '__main__':

    # diff = [13, 6, -3, 7, 9, 8, 0, 11]
    diff = np.subtract(sample1, sample2)
    print(diff)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1)
    diff_length = len(diff)
    alpha = 0.013

    t_statistic = diff_mean / (diff_std / math.sqrt(diff_length))
    t_right = t(diff_length - 1).ppf(1 - alpha)
    print("t_right:", round(t_right, 3))
    print("t_statistic:", round(t_statistic, 3))

    pval = t(diff_length - 1).sf(t_statistic)
    if t_statistic > t_right:
        print("reject null hypothesis, pval is", round(pval, 3))
    else:
        print("not reject null hypothesis, pval is", round(pval, 3))

