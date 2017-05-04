#coding=utf8
'''
Created on 2016-8-30

@author: xuwei

@summary: 
'''
from numpy import array

class MavieNayes(object):
    def __init__(self):
        '''
        @summary: 朴素贝叶斯分类
        '''
        self.BaseData = [
            [u"青绿",u"蜷缩",u"浊响",u"清晰",u"凹陷",u"硬滑",u"是"],
            [u"乌黑",u"蜷缩",u"沉闷",u"清晰",u"凹陷",u"硬滑",u"是"],
            [u"乌黑",u"蜷缩",u"浊响",u"清晰",u"凹陷",u"硬滑",u"是"],
            [u"青绿",u"蜷缩",u"沉闷",u"清晰",u"凹陷",u"硬滑",u"是"],
            [u"浅白",u"蜷缩",u"浊响",u"清晰",u"凹陷",u"硬滑",u"是"],
            [u"青绿",u"稍蜷",u"浊响",u"清晰",u"稍凹",u"软粘",u"是"],
            [u"乌黑",u"稍蜷",u"浊响",u"稍糊",u"稍凹",u"软粘",u"是"],
            [u"乌黑",u"稍蜷",u"浊响",u"清晰",u"稍凹",u"硬滑",u"是"],
            [u"乌黑",u"稍蜷",u"沉闷",u"稍糊",u"稍凹",u"硬滑",u"否"],
            [u"青绿",u"硬挺",u"清脆",u"清晰",u"平坦",u"软粘",u"否"],
            [u"浅白",u"硬挺",u"清脆",u"模糊",u"平坦",u"硬滑",u"否"],
            [u"浅白",u"蜷缩",u"浊响",u"模糊",u"平坦",u"软粘",u"否"],
            [u"青绿",u"稍蜷",u"浊响",u"稍糊",u"凹陷",u"硬滑",u"否"],
            [u"浅白",u"稍蜷",u"沉闷",u"稍糊",u"凹陷",u"硬滑",u"否"],
            [u"乌黑",u"稍蜷",u"浊响",u"清晰",u"稍凹",u"软粘",u"否"],
            [u"浅白",u"蜷缩",u"浊响",u"模糊",u"平坦",u"硬滑",u"否"],
            [u"青绿",u"蜷缩",u"沉闷",u"稍糊",u"稍凹",u"硬滑",u"否"]]
        self.init_training_sample()
    
    def init_training_sample(self):
        '''
        @summary: 加载训练样本
        '''
        self.AttrProb = []
        TrainData = array(self.BaseData)
        Tables = TrainData[:,-1]
        self.calculate_table(Tables)
        AttrNum = TrainData.shape[-1] - 1
        for i in range(AttrNum):
            attrnames = TrainData[:,i]
            attr_prob = self.calculate_attr(attrnames, Tables)
            self.AttrProb.append(attr_prob)
            
    def calculate_table(self,Tables):
        '''
        @summary: 计算标签概率
        P(好瓜=是)    P(好瓜=否)
        '''
        Total = len(Tables)
        self.TableProbs = {}
        tableNumDict = {}     
        for tablename in Tables:
            if tableNumDict.has_key(tablename):
                tableNumDict[tablename] += 1
            else:
                tableNumDict[tablename] = 1
        for tablename,num in tableNumDict.items():
            self.TableProbs[tablename] = float(num)/Total
        
    def calculate_attr(self,attrnames,Tables):
        '''
        @summary: 计算各属性的值对应各标签值的概率（使用拉普拉斯平滑）
        P(attr|好瓜=是) P(attr|好瓜=否)
        '''
        attr_calculate_info = {}
        attr_info = {}
        for i in range(len(attrnames)):
            attrname = attrnames[i]
            tablename = Tables[i]
            if attr_info.has_key(attrname):
                if attr_info[attrname].has_key(tablename):
                    attr_info[attrname][tablename] += 1
                else:
                    attr_info[attrname][tablename] = 1
            else:
                attr_info[attrname] = {tablename:1}
        
        '''计算每一种标签对应的数量'''
        tableNumDict = {}     
        for tablename in Tables:
            if tableNumDict.has_key(tablename):
                tableNumDict[tablename] += 1
            else:
                tableNumDict[tablename] = 1
        '''求出每一种属性值对应每一种标签的概率
        (拉普拉斯平滑：分子+1，分母+属性种类个数)'''
        set_table = set(Tables)
        attr_name_num = len(attr_info)
        for attr_name,tablevalue in attr_info.items():
            attr_calculate_info[attr_name] = {}
            '''如果有些标签的值在某一属性下没有对应的该属性值,添加该标签对应的该属性值的个数 0,以方便计算'''
            if len(tablevalue) < len(set_table):
                for name in set_table:
                    if name not in tablevalue:
                        tablevalue[name] = 0
            for table in tablevalue:
                this_tablenum = tableNumDict.get(table)
                '''该属性值在标该签中的的比例 (拉普拉斯平滑：分子+1，分母+属性种类个数    等于说一共加了：属性种类个数* 1/属性种类个数 = 1)'''
                attr_calculate_info[attr_name][table] = (attr_info[attr_name][table] + 1.0)/(this_tablenum + float(attr_name_num))
        return attr_calculate_info
                
    def calculate_NB(self,attr_list):
        '''
        @summary: 朴素贝叶斯计算并比较，选择概率大的标签返回
        '''
        max_prob = None
        for tablename in self.TableProbs:
            tableProb = self.TableProbs.get(tablename)
            thisProb = tableProb
            for i in range(len(attr_list)):
                attr_name = attr_list[i]
                thisProb *= self.AttrProb[i][attr_name][tablename]
            if not max_prob:
                max_prob = (tablename,thisProb)
            else:
                if max_prob[1] < thisProb:
                    max_prob = (tablename,thisProb)
        return max_prob
            
    def test_interface(self,attr_list):
        '''
        @summary: 测试样本接口，返回对应标签
        '''
        max_prob = self.calculate_NB(attr_list)
        print max_prob[0],max_prob[1]

if __name__ == "__main__":
    attr_list = [u"浅白",u"硬挺",u"清脆",u"模糊",u"平坦",u"硬滑"]
    AA = MavieNayes()
    AA.test_interface(attr_list)
    
