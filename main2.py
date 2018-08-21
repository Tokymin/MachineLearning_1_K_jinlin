import kNN
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

datingDataMat, datingLabels = kNN.file2mtrix('datingTestSet2.txt')  # 从文本文件导入数据
# 在绘图窗口中显示数据的分布图像
# fig = plt.figure()
# # ax = fig.add_subplot(111)  # 111大概是分块的意思
# # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# # plt.show()
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)  # 归一化处理
print(normMat)
print(ranges)
print(minVals)
kNN.datingClassTest()
