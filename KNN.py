#coding=utf8
'''
Created on 2016-5-27

@author: xuwei

@summary: 
'''
from numpy import *  

'''训练样本属性：瓜蒂卷曲度[0-10]，敲声清脆度[0-10]'''
sample_attrbule = [[1,1], [3,2], [7,6], [6,8]]
'''训练样本标签：好瓜，坏瓜'''
sample_lable = ['坏瓜','坏瓜','好瓜','好瓜']

def GetTrainingData():
    '''获取训练样本和训练样本标签''' 
    trainingsamples = array(sample_attrbule)
    traininglables = sample_lable
    return trainingsamples,traininglables

def KNN(TestData,TraininggData):
    trainingsamples,traininglables = TraininggData
    [M,N] = trainingsamples.shape
    print "样本个数：",M,"属性个数：",N
    '''tile(list,(x,y)):将一个y维数组list转为x维；下面就是将一维的TestData转为M个相同的TestData组成的数组'''
    TestDatas = tile(TestData,(M,1))
    '''求数组中的每个元素的距离差(数组之间的计算一一对应)'''
    deffData = TestDatas - trainingsamples
    '''对数组进行评方时，实际是对数组的每个元素进行评'''
    SqDatas = deffData**2
    '''将数组中的每行数据相加(axis=1)，如果是所有数据相加(axis=0),得到两点距离的平方'''
    SqDistance = SqDatas.sum(axis=1)
    '''获取距各个训练样本坐标的距离数组'''
    SqArr = SqDistance**0.5
    SqList = list(SqArr)
    minsq = min(SqList)
    minindex = SqList.index(minsq)
    return traininglables[minindex]
    
if __name__ == "__main__":
    TraininggData = GetTrainingData()
    TestDataList = [[3,3],[5,5],[7,7]]
    for TestData in TestDataList:
        print KNN(TestData,TraininggData)

