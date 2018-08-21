import kNN
from numpy import *

group, labels = kNN.createDataSet()
print(kNN.classify0([1, 1], group, labels, 3))
# 测试tile
# print(tile([0, 0], (4, 1)))
# 测试sum
# a = array([[0, 1, 2], [3, 4, 5]])
# a=mat(a)
# print(a.sum(axis=1))
# 测试shape
# e = eye(4)
# print(e)
# print(e.shape[1])
