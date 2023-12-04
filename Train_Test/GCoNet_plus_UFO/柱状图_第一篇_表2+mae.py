import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab

label = ['Backbone', 'WO/RAAM', 'WO/PDM', 'WO/ASDL', 'PENet-KD']
first = [0.839, 0.842, 0.843, 0.846, 0.856]  # 不同颜色 代表不同指标
secon = [0.918, 0.923, 0.925, 0.930, 0.938]
third = [0.879, 0.886, 0.887, 0.889, 0.902]
four = [0.074, 0.069, 0.068, 0.065, 0.058]  # y
four = [i * 2.5 + 0.7 for i in four]  # x

data = [first, secon, third, four]
date2 = [four]


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

    ax2 = plt.twinx()
    i = 0
    for index, y in enumerate(datas):  # 0 [0.839, 0.842, 0.843, 0.846, 0.856]
        if i == 3:
            print("3")
            ax2.bar(baseline_x + index * bar_span, y, bar_width, zorder=10, color='r')

        else:
            print("012")
            plt.bar(baseline_x + index * bar_span, y, bar_width, zorder=10)
        i = i + 1

    # plt.legend(handles=scatter.legend_elements(num=color)[0], labels=label, loc=4)
    # plt.ylabel('Scores')
    plt.title('NEU RSDDS-AUG')
    # x轴刻度标签位置与x轴刻度一致
    plt.xticks(ticks, labels, fontsize=14)
    plt.rcParams.update({'font.size': 13})
    plt.legend(
        ['S-measure', 'maxE', 'maxF', 'MAE'])

    minor_ticks_top = np.linspace(0.7, 0.95, 21)
    # major_ticks_top = np.linspace(0.08, 0.1, 21)

    plt.yticks(minor_ticks_top)
    # ax2.plt.yticks(major_ticks_top)

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
