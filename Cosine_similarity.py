#coding=utf8
'''
Created on 2016-11-2

@author: xuwei

@summary: 优化后的余弦相似度
基于用户的相似度算法：
    1. 计算所有用户两两之间的相似度
    2. 将与当前用户相似度高的用户喜欢的产品推送给当前用户
    * 因为需要计算每一个用户之间的相似度，当用户量较大时，计算量巨大
    
基于产品的相似度算法：
    1. 通过对比两者产品都有评价的用户评分，计算两者之间的相似度
    2. 如果产品A与产品B相似度较高，向喜欢A的用户推荐B

为了避免分数膨胀，可以使用修正余弦相似度来计算：
    1. 修正余弦相似度和皮尔森相似系数公式相同，但参数不同
    2. 在皮尔森相似系数中 SetX是“用户X”对不同产品的评价，SetY是“用户Y”对不同产品的评价
    3. 在修正余弦相似度中 SetX是不同用户对“产品X”的评价，SetY是不同用户对“产品Y”的评价
'''

import math

def Pearson(SetX,SetY):
    '''
    @summary: 皮尔逊相关系数（Pearson correlation coefficient） 值越大相似度越高  0 < coef <=1
    1. 如果其中一个对象属性为空，则该属性不参与计算
    '''
    AttrNum = len(SetX)
    '''两个属性都不为空的个数'''
    InterNum = len([i for i in range(AttrNum) if SetX[i] and SetY[i]])
    '''求平均值'''
    averageX = float(sum([value for value in SetX if value])/InterNum)
    averageY = float(sum([value for value in SetY if value])/InterNum)
    Add1 = Add2 = Add3 = 0.0
    for i in range(AttrNum):
        if SetX[i] and SetY[i]:
            '''(xi - x_ave)*(yi - y_ave);(xi - x_ave)**2;(yi - y_ave)**2'''
            Add1 += (SetX[i] - averageX)*(SetY[i] - averageY)
            Add2 += math.pow((SetX[i] - averageX),2)
            Add3 += math.pow((SetY[i] - averageY),2)
    distance = Add1/(math.sqrt(Add2)*math.sqrt(Add3))
    return distance
