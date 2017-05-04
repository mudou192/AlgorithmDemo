#coding=utf8
'''
Created on 2016-11-7

@author: xuwei

@summary: K-Means++聚类
K-Means聚类:
    1.随机选取k个元素作为中心点；
    2.根据距离将各个点分配给中心点；
    3.计算新的中心点；
    4.重复2、3，直至满足条件。
 K-Means++聚类:   
    1. 随机选取一个点；
    2. 重复以下步骤，直到选完k个点：
        .计算每个数据点（dp）到各个中心点的距离（D），选取最小的值，记为D(dp)；
        .根据D(dp)的概率来随机选取一个点作为中心点。
    3.根据距离将各个点分配给中心点；
    4.计算新的中心点；
    5.重复3、4，直至满足条件。
    
这里只处理二维数据
在20x20的二维空间里，随机生成一定数量的坐标点，将这些坐标点聚为k类
'''
import random
import math
from matplotlib import pyplot as plt

class KMeans(object):
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        self.TrainingData = [[random.randint(0,20),random.randint(0,20)] for i in range(250)]
        
    def get_nearest_dest(self,PointList,TrainingData):
        '''
        @summary: 计算每个数据点（dp）到各个中心点的距离（D），选取最小的值，记为D(dp)
        '''
        DistList = []
        for data in TrainingData:
            nearest = 1.0
            for Point in PointList:
                dist = math.sqrt((data[0]-Point[0])**2 + (data[1]-Point[1])**2)
                if dist < nearest:
                    nearest = dist
            DistList.append(nearest)
        return DistList
    
    def check_start_point(self,TrainingData, K = 5):
        '''
        @summary: K-Means算法在初始中心点的选择上容易使的结果陷入局部最优，K-Means++改进了起始点的选取过程，其余的和K-Means一致。
        1. 随机选取一个点；
        2. 重复以下步骤，直到选完k个点：
            .计算每个数据点（dp）到各个中心点的距离（D），选取最小的值，记为D(dp)；
            .根据D(dp)的概率来随机选取一个点作为中心点。
        '''
        PointList = []
        #1. 在数据集中随机选取一个点
        Point = (random.choice(TrainingData))
        PointList.append(Point)
        for i in range(K-1):
            #2. 计算每个点到最近的中心点的距离
            DistList = self.get_nearest_dest(PointList, TrainingData)
            #3. 将距离转换为权重
            distadd = sum(DistList)
            DistList = [dist/distadd for dist in DistList]
            #4. 随机选择中心点(距离较大的选择的概率就大)
            total = 0
            index = 0
            Pointer = random.random()
            while total < Pointer:
                total += DistList[index]
                index += 1
            PointList.append(TrainingData[index])
        return PointList
    
    def cluster_point(self,TrainingData,PointList):
        '''
        @summary: 聚类，将数据分配给距离最近的中心点
        '''
        ClusterSet = [[] for i in range(len(PointList))]
        for Data in TrainingData:
            ret_list = []
            for Point in PointList:
                Dist = math.sqrt((Data[0]-Point[0])**2 + (Data[1]-Point[1])**2)
                ret_list.append(Dist)
            index = ret_list.index(min(ret_list))
            ClusterSet[index].append(Data)
        return ClusterSet
    
    def update_point(self,ClusterSet,PointList):
        '''
        @summary: 更新中心点
        '''
        i = 0
        for SetData in ClusterSet:
            if SetData:
                PointList[i][0] = sum([data[0] for data in SetData])/float(len(SetData))
                PointList[i][1] = sum([data[1] for data in SetData])/float(len(SetData))
            else:
                PointList[i] = (random.choice(self.TrainingData))
            i += 1
        return PointList
    
    def show(self,ClusterSet):
        '''
        @summary: 将这些点画出来
        '''
        colours = ['r','b','y','m','c']
        i = 0
        x_list = []
        y_list = []
        colour_list = []
        for Cluster in ClusterSet:
            x_list.extend([p[0] for p in Cluster])
            y_list.extend([p[1] for p in Cluster])
            colour_list.extend([colours[i]]*len(Cluster))
            i += 1
        plt.scatter(x_list,y_list,c=colour_list,s=45)
        plt.show()
    
    def main(self,times = 100):
        '''
        @summary: 主方法
        '''
        PointList = self.check_start_point(self.TrainingData, K = 5)
        num = 0
        while num < times:
            ClusterSet = self.cluster_point(self.TrainingData, PointList)
            PointList = self.update_point(ClusterSet, PointList)
            num += 1
        self.show(ClusterSet)
        
if __name__ == "__main__":
    A = KMeans()
    A.main()
