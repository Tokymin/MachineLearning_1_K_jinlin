from numpy import *
import operator


# operator是运算符模块


def createDataSet():
    """该函数用于创建数据集合标签"""
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # .shape 为矩阵的长度 在4×2的矩阵c, c.shape[1] 为第一维的长度，c.shape[0] 为第二维的长度。
    """tile(A,reps)字面意思：将A矩阵（其他数据类型）重复reps次 
    例如tile((1,2,3),3)==>array([1, 2, 3, 1, 2, 3, 1, 2, 3])
     如果是b=[1,3,5]
        tile(b,[2,3])
        array([[1, 3, 5, 1, 3, 5, 1, 3, 5],
       [1, 3, 5, 1, 3, 5, 1, 3, 5]]) 2指的是重复后矩阵的行数，而3指的是重复次数
        就如这里一样
    """
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # title([0,0],(4,1)) ==>[[0 0],[0 0],[0 0],[0 0]]
    sqDiffMat = diffMat ** 2  # 计算距离
    """
    平时用的sum应该是默认的axis=0 就是普通的相加 
    axis=1以后就是将一个矩阵的每一行向量相加
    """
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def file2mtrix(filename):
    """将文本文件转化NumPy的解析"""
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  # 转化为str？去掉空格 Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）或字符序列。注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  # 表示列表中的最后一列元素，用于存在label标签数组中，转为int不然是字符串
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """将当前的数据进行归一化处理"""
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """用于记录不符合要求的数据，最后计算一个错误率"""
    hoRatio = 0.10
    datingDataMat, datingLabels = file2mtrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with:%d,the real answer is :%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print("the total error rate is :%f " % (errorCount / float(numTestVecs)))
