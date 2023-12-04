import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
label = ['S2MA', 'EDR', 'BBS', 'HAI', 'EMI', 'DAC', 'MC', 'DRER', 'LENO', 'CLA', 'HRT', 'ICON', 'PENet\n-Student', 'PENet\n-KD', 'PENet\n-Teacher']
first = [0.775, 0.811, 0.828, 0.718, 0.800, 0.824, 0.822, 0.844, 0.817, 0.835, 0.788, 0.843, 0.846, 0.856, 0.859]  # 不同颜色 代表不同指标
secon = [0.864, 0.893, 0.909, 0.829, 0.876, 0.911, 0.915, 0.933, 0.900, 0.920, 0.873, 0.925, 0.930, 0.938, 0.940]
third = [0.817, 0.850, 0.867, 0.803, 0.850, 0.875, 0.865, 0.891, 0.857, 0.878, 0.824, 0.884, 0.889, 0.902, 0.905]

data = [first, secon, third]


def create_multi_bars(labels, datas, tick_step=1, group_gap=0.2, bar_gap=0.5):
    '''
    labels : x轴坐标标签序列
    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致
    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。
    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠
    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠
    '''

    # ticks为x轴刻度
    ticks = np.arange(len(labels)) * tick_step
    # group_num为数据的组数，即每组柱子的柱子个数
    group_num = len(datas)
    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。
    group_width = (tick_step - group_gap)
    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和
    bar_span = group_width / group_num
    # bar_width为每个柱子的实际宽度
    bar_width = bar_span - bar_gap
    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可
    baseline_x = ticks - (group_width - bar_span) / 2
    for index, y in enumerate(datas):
        plt.bar(baseline_x + index * bar_span, y, bar_width, zorder = 10)
    # plt.legend(handles=scatter.legend_elements(num=color)[0], labels=label, loc=4)
    # plt.ylabel('Scores')
    plt.title('NEU RSDDS-AUG')
    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, labels, fontsize=14)
    plt.rcParams.update({'font.size':20})
    plt.legend(
        ['S-measure', 'maxE', 'maxF'])


    # plt.axhline(0.938, linestyle="--")
    # plt.grid(linestyle='--', zorder = 0)
    # major_ticks_top = np.linspace(0.7, 0.9, 10)
    minor_ticks_top = np.linspace(0.7, 0.95, 21)
    #
    # plt.yticks(major_ticks_top)
    # # plt.set_xticks(minor_ticks_top, minor=True)
    plt.yticks(minor_ticks_top)
    plt.grid(which='minor', alpha=0.6)
    plt.grid(
        ls='--',  # 样式
        lw=1,
        c='grey',
        axis='y',  # 让哪个轴显示网格线，这里x轴
    )
    # plt.grid(which='major', alpha=1)
    plt.show()


create_multi_bars(label, data, bar_gap=0.1)
