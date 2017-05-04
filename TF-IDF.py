#coding=utf8
'''
Created on 2016-9-18

@author: xuwei

@summary: TF-IDF 算法
'''
import jieba,os,json
from math import log

class TFIDF(object):
    def __init__(self):
        self.DocsWord = []
    
    def initCorpusData(self,file_path):
        '''
        @summary: 初始化语料库数据
        '''
        filelist = os.listdir(file_path)
        for filename in filelist:
            if "000" in filename:
                filename = os.path.join(file_path,filename)
                print "Init File:",filename
                WordDict,AllWordNum = self.SplitWord(filename)
                self.DocsWord.append(WordDict)
        CorpusData = json.dumps(self.DocsWord)
        with open('CorpusData','wb') as fp:
            fp.write(CorpusData)
            
    def loadCorpusData(self):
        '''
        @summary: 加载语料库数据
        '''
        with open('CorpusData','wb') as fp:
            CorpusData = fp.read()
        self.DocsWord = json.loads(CorpusData)
            
    
    def SplitWord(self,filename):
        '''
        @summary: 对文档进行分词
        '''
        WordDict = {}
        with open(filename,'rb') as fp:
            Content = fp.read()
        WordList = jieba.cut(Content)
        AllWordNum = 0
        for word in WordList:
            AllWordNum += 1
            if word in WordDict:
                WordDict[word] += 1
            else:
                WordDict[word] = 1
        return WordDict,AllWordNum
    
    
    def TermFrequency(self,NowWord,NowWordDict,NowAllWordNum):
        '''
        @summary: 步骤 1    计算词频
        TF(wi) = C(wi)/∑C(wn) 文章中当前词出现的次数/文章中的总词数
        '''
        WordNum = NowWordDict.get(NowWord)
        TF = WordNum/NowAllWordNum
        return TF
    
    def TermInverDocFreq(self,NowWord):
        '''
        @summary: 步骤 2    计算逆文档频率
        IDF = log(D / (docs(w,D) + 1)) 文档总数n与词w所出现文件数docs(w, D)比值的对数
        '''
        HasWordNum = 0
        for Doc in self.DocsWord:
            if Doc.get(NowWord):
                HasWordNum += 1
        IDF = log(float(len(self.DocsWord))/(HasWordNum + 1))
        return IDF
    
    def TermTFIDF(self,TF,IDF):
        '''
        @summary: 步骤 3    计算TF-IDF
        TF-IDF = TF * IDF
        '''
        TF_IDF = TF * IDF
        return TF_IDF
    
    def main(self,filename,MaxNum = 50):
        '''
        @summary: 主方法
        '''
        WordInfoList = []
        NowWordDict,NowAllWordNum = self.SplitWord(filename)
        for NowWord in NowWordDict:
            TF = self.TermFrequency(NowWord, NowWordDict, NowAllWordNum)
            IDF = self.TermInverDocFreq(NowWord)
            TF_IDF = self.TermTFIDF(TF, IDF)
            WordInfoList.append((NowWord,TF_IDF))
        WordInfoList.sort(key = lambda x: -x[1])
        TopWords = WordInfoList[0:MaxNum]
        return TopWords
        
if __name__ == "__main__":
    AA = TFIDF()
    CorpusPath = r"C:\Users\Administrator\Desktop\linshi"
    TestFile = r"test_data"
    AA.initCorpusData(CorpusPath)
    TopWords = AA.main(TestFile, MaxNum = 100)
    for word,TF_IDF in TopWords:
        print "%15s:%f"%(word,TF_IDF)
        
