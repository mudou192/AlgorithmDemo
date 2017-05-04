#coding=utf8
'''
Created on 2016-11-1

@author: xuwei

@summary: 闵可夫斯基距离(Minkowski distance)
'''
import math,numpy

def Manhattan(SetX,SetY):
    '''
    @summary: 曼哈顿距离（Manhattan distance） 
    1. 如果其中一个对象属性为空，则该属性不参与计算
    2. ∑|Xi - Yi|    X:对象X， Y:对象Y， i:第i个属性
    '''
    distance = 0.0
    AttrNum = len(SetX)
    for i in range(AttrNum):
        if SetX[i] and SetY[i]:
            distance += abs(SetX[i] - SetY[i])
    return distance

def Euclidean(SetX,SetY):
    '''
    @summary: 欧几里得距离（Euclidean distance 勾股定理） 
    1. 如果其中一个对象属性为空，则该属性不参与计算
    2. √￣∑(Xi - Yi)**2
    '''
    distance = 0.0
    AttrNum = len(SetX)
    for i in range(AttrNum):
        if SetX[i] and SetY[i]:
            distance += math.pow((SetX[i] - SetY[i]),2)
    return math.sqrt(distance)

def Minkowski(SetX,SetY,r):
    '''
    @summary: 闵可夫斯基距离(Minkowski distance)
    1. 如果其中一个对象属性为空，则该属性不参与计算
    2. (∑|Xi - Yi|**r)**1/r
    r = 1 该公式即曼哈顿距离
    r = 2 该公式即欧几里得距离
    r = ∞ 极大距离
    '''
    distance = 0.0
    AttrNum = len(SetX)
    for i in range(AttrNum):
        if SetX[i] and SetY[i]:
            distance += math.pow((SetX[i] - SetY[i]),r)
    return math.pow(distance,1/r)

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

def Cosine(SetX,SetY):
    '''
    @summary: 余弦相似度（Cosine similarity） 值越大相似度越高  0 < coef <=1
    1. 如果其中一个对象属性为空，则该属性不参与计算
    cos(x,y) = x*y/(||x||*||y||)
    ||x|| = √￣∑(Xi)**2
    '''
    AttrNum = len(SetX)
    for i in range(AttrNum):
        '''将不能参与计算的值由None置为0，不影响计算数值'''
        if not SetX[i] or not SetY[i]:
            SetX[i] = SetY[i] = 0
    '''x*y; ||x||; ||y||'''
    VProduct = sum(numpy.array(SetX) * numpy.array(SetY))
    VModuleX = math.sqrt(sum([value**2 for value in SetX]))
    VModuleY = math.sqrt(sum([value**2 for value in SetY]))
    distance = VProduct/(VModuleX*VModuleY)
    return distance
    
if __name__ == "__main__":
    SetX = [1,2,3,4,5]
    SetY = [1,2,3,4,5]
    print "曼哈顿距离:\t",
    print Manhattan(SetX,SetY)
    print "欧几里得距离:\t",
    print Euclidean(SetX, SetY)
    print "余弦相似度:\t",
    print Cosine(SetX, SetY)
    print "-----------------------------------"
    SetX = [1,2,3,4,5]
    SetY = [1.1,1.2,1.3,1.4,1.5]
    print "皮尔逊相关系数:\t",
    print Pearson(SetX, SetY)
    
