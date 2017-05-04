#coding=utf8
'''
Created on 2016-6-1
@author: xuwei
@summary: 参考 http://www.hankcs.com/ml/decision-tree.html
'''
from math import log
def calculateEntropy(samplesSet):
    '''
    @summary: 计算数据集的信息熵
    @samplesSet(样本集合)：
        [([x0,x1,x2,x3],True),([x0,x1,x2,x3],False),([x0,x1,x2,x3],True)]
    1、遍历每个样本，统计标签的频次
    2、得出每个标签在数据集中的的比例
    3、根据各个标签的比例求出信息熵
    '''
    samplesnum = len(samplesSet)
    lablesdict = {}
    for samplesinfo in samplesSet:
        lable = samplesinfo[1]
        if lablesdict.get(lable):
            lablesdict[lable] += 1
        else:
            lablesdict[lable] = 1
    entropy = 0.0
    for lable,times in lablesdict.items():
        times = float(times)
        probability = times/samplesnum
        entropy -= probability*log(probability,2)
    return entropy
def splitSamplesSet(samplesSet,index,attribute):
    '''
    @summary: 获取index维度属性为attribute的数据集合
    @param index: 数据的维度
    @param attribute: 数据的属性
    '''
    retSet = []
    for samples in samplesSet:
        attrSet = samples[0]
        if attrSet[index] == attribute:
            retSet.append(samples)
    return retSet
def calculateConditionEntropy(samplesSet,index,attrSet):
    '''
    @summary: 计算条件熵（当数据集合第index维属性值被获知时的信息熵）
    @param index: 数据的维度(条件:已知维度属性),x0: dim = 0,x1 : dim = 1...
    @param attrSet: 该维度所有属性的集合
    1、遍历当前维度的属性值
    2、获取每个属性对应的样本
    3、求出当前属性集合占数据集的比例，并求出当前属性对应的数据集的信息熵
    4、将各个属性集合对应的信息熵乘以比例并相加——>条件熵
    '''
    ConEntropy = 0.0
    for attribute in attrSet:
        retSet = splitSamplesSet(samplesSet,index,attribute)
        entropy = calculateEntropy(retSet)
        prob = len(retSet)/float(len(samplesSet))
        ConEntropy += prob * entropy
    return ConEntropy
def calculateGain(samplesSet, index, oldEntropy):
    '''
    @summary: 计算信息增益(熵的减少，即不确定性的减少：原信息熵 - 当前条件熵)
    @param index: 数据的维度
    @param oldEntropy: 原信息熵
    '''
    print index
    attrList = [samples[0][index] for samples in samplesSet]
    print attrList
    attrSet = set(attrList)
    newEntropy = calculateConditionEntropy(samplesSet,index,attrSet)
    return oldEntropy - newEntropy
def featureSelectionByID3(samplesSet):
    '''
    @summary: ID3 特征选择算法(遍历属性集合的索引，计算信息增益，选出信息增益最大值对应的索引作为当前最优节点)
    '''
    attrLength = len(samplesSet[0][0])
    baseEntropy =  calculateEntropy(samplesSet)
    bestGain = 0.0
    bestIndex = -1
    for index in range(attrLength):
        retGain = calculateGain(samplesSet, index, baseEntropy)
        if retGain > bestGain:
            bestGain = retGain
            bestIndex = index
    return bestIndex
def createTree(samplesSet,attrNameSet):
    '''
    @summary: 创建决策树
    1、通过特征选择，选出最优节点的索引
    2、将该索引对应的类别名称添加到决策树中，并该索引对应的的数据从数据集中剔除
    3、将剔除后的的数据再次递归处理，直到数据集长度为0
    '''
    lableList = [samples[-1] for samples in samplesSet]
    if len(set(lableList)) == 1:
        '''如果标签只剩一种时，证明当前条件只有一种结果，返回当前标签'''
        return lableList[0]
    bestIndex = featureSelectionByID3(samplesSet)
    print bestIndex
    attrName = attrNameSet[bestIndex]
    retTree = {attrName:{}}
    del(attrNameSet[bestIndex])
    '''获取最优属性集合'''
    bestAttrList = [samples[0][bestIndex] for samples in samplesSet]
    bestAttrSet = set(bestAttrList)
    for attr in bestAttrSet:
        retlable = lableList[:]
        retsamplesSet = splitSamplesSet(samplesSet,bestIndex,attr)
        print "retsamplesSet:",retsamplesSet
        print "attrName:",attrName
        print "attr:",attr
        value = createTree(retsamplesSet,retlable)
        print "value:",value
        retTree[attrName][attr] = value
    return retTree
dataSet = [([u'青年', u'否', u'否', u'一般'], False),
               ([u'青年', u'否', u'否', u'好'], False),
               ([u'青年', u'是', u'否', u'好'], True),
               ([u'青年', u'是', u'是', u'一般'], True),
               ([u'青年', u'否', u'否', u'一般'], False),
               ([u'中年', u'否', u'否', u'一般'], False),
               ([u'中年', u'否', u'否', u'好'], False),
               ([u'中年', u'是', u'是', u'好'], True),
               ([u'中年', u'否', u'是', u'非常好'], True),
               ([u'中年', u'否', u'是', u'非常好'], True),
               ([u'老年', u'否', u'是', u'非常好'], True),
               ([u'老年', u'否', u'是', u'好'], True),
               ([u'老年', u'是', u'否', u'好'], True),
               ([u'老年', u'是', u'否', u'非常好'], True),
               ([u'老年', u'否', u'否', u'一般'], False),
               ]
attrNameSet = [u'年龄', u'有工作', u'有房子', u'信贷情况']
if __name__ == "__main__":
    print createTree(dataSet,attrNameSet)
