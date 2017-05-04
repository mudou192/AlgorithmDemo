#coding=utf8
'''
Created on 2016-11-2

@author: xuwei

@summary: Slope One算法,支持一个用户对多个商品的评分预测
步骤：
    1. 首先需要计算出两两物品之间的差值（可以在夜间批量计算）
    2. 进行预测
    
User/Product    P1    P2    P3
U1              4     3     4
U2              5     2     ?
U3              ?    3.5    4
U4              5     ?     3



'''
import numpy

test_data = [[4,3,4],
             [5,2,None],
             [None,3.5,4],
             [5,None,3]]


def SlopeOne(TestData):
    '''
    @summary: Slope One算法
    ∑( (ui - uj)/card(Sj,i(X)) )
    card(Sj,i(X))则表示同时评价过物品j和i的用户数
    uj-ui表示用户对j的评分减去对i的评分
    '''
    '''第一步：∑( (ui - uj)/card(Sj,i(X)) )'''
    TestData = numpy.array(TestData)
    U_num,P_num = numpy.shape(TestData)
    diffList = []
    '''各商品的差值为P_num*P_num的矩阵，虽然有重复，但方便后面取值计算'''
    for i in range(P_num):
        ret_diff_list = []
        for j in range(P_num):
            Card = 0
            diff = 0.0
            for n in range(U_num):
                '''1. 找出对Pi，Pj都评论的人的人数和差值集的和'''
                UserScores = TestData[n]
                if UserScores[i] is None or UserScores[j] is None:
                    continue
                else:
                    diff += UserScores[i] - UserScores[j]
                    Card += 1
            '''2. 差值集的和/评论人数 = 两物品之间的差值'''
            dev = diff/Card
            ret_diff_list.append((dev,Card))
        diffList.append(ret_diff_list)
    
    '''第二步：P^WS1(u)j = ∑((dev_ij + u_i)C_ij) / ∑C_ij 其中     dev_ij：i和j的差值；     u_i：当前用户对该物品的评分；     C_ij：card(Sj,i(X))'''
    '''P^WS1(u)j表示我们将预测用户u对物品j的评分'''
    for n in range(U_num):
        UserScores = TestData[n]
        NoneIndexs = []
        j = None
        for i in range(P_num):
            if UserScores[i] is None:
                j = i
                NoneIndexs.append(j)
        for j in NoneIndexs:
            mole = 0.0
            deno = 0.0
            for i in range(P_num):
                if i in NoneIndexs:
                    continue
                '''(dev_ij + u_i)C_ij'''
                dev,Card = diffList[j][i]
                user = UserScores[i]
                mole += (dev + user)*Card
                deno += Card
            EstimateValue = mole/deno
            UserScores[j] = EstimateValue
    return TestData

if __name__ == "__main__":
    print SlopeOne(test_data)
