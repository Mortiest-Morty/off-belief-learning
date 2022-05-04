## -*- coding: utf-8 -*-
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import numpy as np

# data = [23.85, 24.05, 24.15]
# labels = ["DFT", "SPARTA", "RLSearch"]
# err = [0.043, 0.043, 0.043]
# data = [23.85, 23.40]
# labels = ["DFT", "DFT-rollout"]
# err = [0.043, 0.043]
data = [23.85, 23.97, 23.32]
labels = ["DFT", "DFT-exact", "DFT-nonbft"]
err = [0.043, 0.043, 0.043]

color = []
for label in labels:
    if label == "DFT":
        color.append("blue")
    elif label == "SPARTA":
        color.append("orange")
    elif label == "DFT-rollout":
        color.append("red")
    elif label == "DFT-exact":
        color.append("pink")
    elif label == "DFT-nonbft":
        color.append("brown")
    else:
        color.append("green")

err_prams = dict(ecolor="gray")
keys = np.arange(len(data))
bar1 = plt.bar(keys, data, width=0.5, yerr=err, error_kw=err_prams, color=color, alpha=0.6)
# plt.axhline(y=0, color="black")
plt.ylabel("average scores")
plt.xticks(ticks=keys, labels=labels, fontsize=10)


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        print(height)
        height2 = height + 0.05 if height >0 else height - 0.05
        plt.text(rect.get_x() + rect.get_width() / 2., height2,
                 '{:2}'.format(height),
                 ha='center', va='bottom')


autolabel(bar1)
#plt.xticks(rotation=-15)
plt.ylim((20, 25))
#plt.show()
plt.savefig("search_result3.png", dpi=500)