#!usr/bin/env python3.7
# coding:utf-8
"""
@author: zgw
@file: kNN-dating-example.py
@date: 2019/11/12 22:16
@description streamlit kNN改进约会网站的配对效果
"""
import streamlit as st
from numpy import *
import matplotlib.pyplot as plt
import operator
import numpy as np


def open_file(filename):
    """
    打开文件
    :param filename: 文件名称
    :return: 文件内容list
    """
    with open(filename) as fr:
        return fr.readlines()


@st.cache
def file2matrix(filename):
    """
    将文本记录转换为 NumPy
    :param filename: 文件名称
    :return: numpy矩阵
    """
    fr = open_file(filename)
    # 获取文件中行数
    numberOfLines = len(fr)
    # 创建返回的numpy矩阵
    returnMat = zeros((numberOfLines, 3))
    # 解析文件数据保存到列表中
    classLabelVector = []
    index = 0
    for line in fr:
        # str.strip([chars]) --返回已移除字符串头尾指定字符（默认为空格或换行符）所生成的新字符串
        line = line.strip()
        # 以 '\t' 切割字符串
        listFromLine = line.split('\t')
        # 每列的属性数据
        returnMat[index, :] = listFromLine[0:3]
        # 每列的类别数据，就是 label 标签数据
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    # 返回数据矩阵returnMat和对应的类别classLabelVector
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
     归一化特征值，消除特征之间量级不同导致的影响
    :param dataSet: 数据集
    :return:归一化后的数据集 normDataSet. ranges和minVals即范围和最小值
    """
    # 计算每列的最小值和最大值，dataSet.min(0)参数0使得函数从列中选取最小值而不是选取当前行的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 计算可能的取值范围 1*3
    ranges = maxVals - minVals
    # 1000*3
    normDataSet = zeros(shape(dataSet))
    # shape[0]读取矩阵的第一维长度
    m = dataSet.shape[0]
    # 生成与最小值之差组成的矩阵。tile()将变量内容复制成输入矩阵同大小的矩阵
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 将最小值之差除以范围组成矩阵
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 特征值相除
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    """
    k-近邻算法
    :param inX: 用于分类的输入向量
    :param dataSet: 输入的训练样本集
    :param labels: 标签向量
    :param k: 表示用于选择最近邻居的数目
    :return: 前k个点中出现频率最高的那个分类，作为当前点的预测分类
    """
    dataSetSize = dataSet.shape[0]
    # 距离度量 度量公式为欧氏距离公式
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 将距离排序：从小到大
    sortedDistIndicies = distances.argsort()
    # 选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def datingClassTest(k):
    """
    对约会网站的测试
    :return: 错误率和错误数
    """
    # 设置测试数据的的一个比例（训练数据集比例=1-hoRatio）
    hoRatio = 0.1  # 测试范围,一部分测试一部分作为样本
    # 从文件中加载数据
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # m 表示数据的行数，即矩阵的第一维
    m = normMat.shape[0]
    # 设置测试的样本数量， numTestVecs:m表示训练样本的数量
    numTestVecs = int(m * hoRatio)
    st.markdown(f'k值为{k},测试的样本数量numTestVecs={numTestVecs}')
    errorCount = 0.0
    for i in range(numTestVecs):
        # 对数据测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], k)
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
            st.markdown(f"预测分类为: {classifierResult},真正的分类为: {datingLabels[i]}")
    st.markdown(f"错误率为: {errorCount / float(numTestVecs)}")
    st.markdown(f"错误个数: {errorCount}")
    return errorCount, errorCount / float(numTestVecs)


def classifyPerson(ffMiles, percentTats, iceCream, k):
    """
    约会网站预测函数
    :return: 预测结果
    """
    resultList = ['not at all不喜欢的人', 'in small doses魅力一般的人', 'in large doses极具魅力的人']
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream, ])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, k)
    return resultList[classifierResult - 1]


def play_default_scatter_plots(datingDataMat):
    # 显示中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 设置正常显示符号，解决保存图像是符号’-‘显示方块
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_title('散点分析图:没有样本类别标签的约会数据散点图，难以辨别图中点究竟属于哪个样本分类')
    ax.set_xlabel('玩视频游戏所耗时间百分比')
    ax.set_ylabel('每周消费的冰淇淋公升数')
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    st.pyplot()


def play_scatter_plots_with_different_size_and_color(datingDataMat, datingLabels):
    # 显示中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 设置正常显示符号，解决保存图像是符号’-‘显示方块
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_title('散点分析图:利用颜色和尺寸标识数据点的属性类别，基本可看出数据点所属三个样本分类的区域轮廓')
    ax.set_xlabel('玩视频游戏所耗时间百分比')
    ax.set_ylabel('每周消费的冰淇淋公升数')
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    st.pyplot()


def play_scatter_plots_with_different_features(datingDataMat, datingLabels):
    # 显示中文
    plt.rcParams['font.sans-serif'] = 'SimHei'
    # 设置正常显示符号，解决保存图像是符号’-‘显示方块
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_title('飞行常客里程数和玩视频游戏所占百分比的散点分析图，区分数据点从属的类别')
    ax.set_xlabel('每年获得的飞行常客里程数')
    ax.set_ylabel('玩视频游戏所耗时间百分比')
    datingLabels = np.array(datingLabels)
    # datingLabels为1的索引
    index_1 = np.where(datingLabels == 1)
    type_1 = ax.scatter(datingDataMat[index_1, 0], datingDataMat[index_1, 1], color='r')
    index_2 = np.where(datingLabels == 2)
    type_2 = ax.scatter(datingDataMat[index_2, 0], datingDataMat[index_2, 1], color='g')
    index_3 = np.where(datingLabels == 3)
    type_3 = ax.scatter(datingDataMat[index_3, 0], datingDataMat[index_3, 1], color='b')
    ax.legend([type_1, type_2, type_3], ['不喜欢', '魅力一般', '极具魅力'], loc=2)
    st.pyplot()


if __name__ == '__main__':
    st.sidebar.header("机器学习实战")
    st.sidebar.markdown("---")
    btn_default = st.sidebar.button("1.默认情况")
    btn_different_size = st.sidebar.button("2.使用不同颜色大小")
    btn_different_features = st.sidebar.button("3.使用另外两个特征")
    st.sidebar.markdown("---")
    k = st.sidebar.slider("选取k值", 1, 10, 3)
    st.sidebar.markdown("当前k值为：{}".format(k))
    btn_test = st.sidebar.button("选取k值后，训练模型")
    st.sidebar.markdown("---")
    btn_classify_person = st.sidebar.button("约会网站结果预测")

    st.header('案例：使用k-近邻算法改进约会网站的配对效果')
    ffMiles = st.number_input("每年获得的飞行常客里程数?")
    percentTats = st.number_input("玩视频游戏所耗费时间百分比?")
    iceCream = st.number_input("每周消费的冰淇淋公升数?",value=0.0)
    filename = 'data/datingTestSet2.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    if btn_default:
        st.subheader("散点分析图:没有样本类别标签的约会数据散点图，难以辨别图中点究竟属于哪个样本分类")
        play_default_scatter_plots(datingDataMat)

    if btn_different_size:
        st.subheader("散点分析图:利用颜色和尺寸标识数据点的属性类别，基本可看出数据点所属三个样本分类的区域轮廓")
        play_scatter_plots_with_different_size_and_color(datingDataMat, datingLabels)

    if btn_different_features:
        st.subheader("飞行常客里程数和玩视频游戏所占百分比的散点分析图，区分数据点从属的类别")
        play_scatter_plots_with_different_features(datingDataMat, datingLabels)

    if btn_test:
        st.subheader("选取k值后进行训练")
        with st.spinner('正在训练...'):
            datingClassTest(k)
        st.success("训练完成")
    if btn_classify_person:
        st.subheader("预测结果")
        with st.spinner('开始预测...'):
            result = classifyPerson(ffMiles, percentTats, iceCream, k)
        st.success(f"预测完成,预测结果为：{result}")
