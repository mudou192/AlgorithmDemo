#coding=utf8
'''
Created on 2016-9-13

@author: xuwei

@summary: Bi-Gram

定义：如果一个词的出现仅仅依赖于它前面出现的一个词，我们就称之为Bi-Gram
参考：http://blog.csdn.net/baimafujinji/article/details/51281816

马尔可夫链:
    在一个随机过程中，如果事件发生概率在t时刻所处的状态为已知时，它在t + 1时刻只与t时刻的状态有关，而与之前所处的状态无关，则称该过程具有马尔可夫性。
    时间和状态都是离散的马尔可夫过程称为马尔可夫链
'''
import json

class BiGram(object):
    def __init__(self):
        pass
    
    def init_modle(self):
        '''
        @summary: 初始化模型
        '''
        print "正在加载训练数据..."
        ContnetList = self.getTextContent()
        print len(ContnetList)
        print "正在生成数据结构..."
        CandidateWord = self.load_word_list(ContnetList)
        print "正在计算数据概率..."
        self.CandidateWord = self.ConditionProbability(CandidateWord)
        print "正在持久化训练结果..."
        self.persistence()
    
    def getTextContent(self):
        '''
        @summary: 加载训练文本
        '''
        with open("splitWord",'rb') as fp:
            Content = fp.read()
        try:
            Content = Content.decode('utf8')
        except:pass
        Content = re.sub('\s+',' ',Content)
        ContnetList = Content.split(u" ")
        return ContnetList
    
    def persistence(self):
        '''
        @summary: 持久化
        '''
        jsonString = json.dumps(self.CandidateWord)
        with open('persistence.json','wb') as fp:
            fp.write(jsonString)
        
    def loadLocalData(self):
        '''
        @summary: 加载持久化数据模型
        '''
        with open('persistence.json','rb') as fp:
            Content = fp.read()
        self.CandidateWord = json.loads(Content)
    
    def load_word_list(self,ContnetList):
        '''
        @summary: 加载数据，剔除数据中的标点符号和垃圾数据，先使用分词工具先分词（这里要识别新词，比如公司名称等）
        @param ContnetList: 分词后的列表，为了节省内存以“,”断句
        '''
        CandidateWord = {}
        length = len(ContnetList)
        for i in range(length-1):
            if i % 10000 == 0:
                print "已完成 :" + str(i/float(length)*100) + " %"
            word = ContnetList[i]
            if not word or word == u"," or word == u" ":
                continue
            
            nextword = ContnetList[i+1]
            
            if word in CandidateWord:
                CandidateWord[word][0] += 1
                
                if nextword and nextword != u"," and nextword != u" ":
                    if nextword in CandidateWord[word][1]:
                        CandidateWord[word][1][nextword][0] += 1
                    else:
                        CandidateWord[word][1][nextword] = [1,0]
            else:
                if nextword and nextword != u"," and nextword != u" ":
                    nextworddict = {ContnetList[i+1]:[1,0]}     #ContnetList[i-1] 上一个的字词，(1,0)出现的次数和概率
                else:
                    nextworddict = {}
                CandidateWord[word] =[1,                #出现次数
                                     nextworddict       #候选词组上一个字词的列表
                                     ]
        return CandidateWord
                
    
    def ConditionProbability(self,CandidateWord):
        '''
        @summary: 计算条件概率:P(wi|wi-1) = C(wi-1wi)/C(wi-1)
        wi 当前字词在全文中出现的次数，wi-1 上一个词在全文中出现的次数,wi-1wi wi-1和wi的组成的词组
        '''
        num = 0
        length = len(CandidateWord)
        for word in CandidateWord:
            num += 1
            if num % 10000 == 0:
                print "已完成 :" + str(num/float(length)*100) + " %"
            nextworddict = CandidateWord[word][1]
            for nextword in nextworddict:
                phraseNum = nextworddict[nextword][0]
                nextWordNum = CandidateWord[nextword][0]
                CandidateWord[word][1][nextword][1] = float(phraseNum)/nextWordNum
        return CandidateWord
    
    def GetMostLikelyWord(self,NowWord,MaxNum = 3, Times = 2):
        '''
        @summary: 获取最优可能的下一个字词
        @param NowWord: 当前字词
        @param MaxNum: 最多返回几个
        @param Times: 最少出现次数（很多词只出现一两次但计算的概率会很高，虽然也有一些少见词语，但是更多的是测试集的问题）
        '''
        WordInfo = self.CandidateWord.get(NowWord)
        if not WordInfo:
            return None
        NextWordDict = WordInfo[1]
        WordList = []
        for nextWord in NextWordDict:
            Num,Pro = NextWordDict.get(nextWord)
            if Num <= Times:
                continue
            WordList.append((nextWord,Num,Pro))
        WordList.sort(key = lambda x: -x[2])
        RetList = WordList[0:MaxNum]
        ResultWords = [i[0] for i in RetList]
        return ResultWords
    
    def SentenceProbability(self,WordList):
        '''
        @summary: 计算一条语句的概率:P(w1,w2,w3...wn) = ∏P(wi|wi-1)
        @param WordList: 一个语句分割后的列表：
            :我爱北京天安门 ==> 我,爱,北,京,天,安,门 
            :也可以是  我,爱,北京,天安门;取决于训练的数据是按字拆分还是按词拆分(看需求)
        Add-one:平滑方式，规定任何一个词在n-gram样本中至少出现一次
        '''
        totalPro = 1.0
        for i in range(len(WordList)):
            if i < len(WordList) - 1:
                nowword = WordList[i]
                nextword = WordList[i+1]
                wordInfo = self.CandidateWord.get(nowword)
                if not wordInfo:
                    pro = 1.0/len(self.CandidateWord)
                    totalPro *= pro
                    continue
                NextWordDict = wordInfo[1]
                NextWordInfo = NextWordDict.get(nextword)
                if not NextWordInfo:
                    pro = 1.0/len(self.CandidateWord)
                    totalPro *= pro
                    continue
                pro = NextWordInfo[1]
                totalPro *= pro
        return totalPro

import jieba,re

unused_words=u" ，。：；“‘”【】『』|=+-－——（）*&……%￥#@＆！~·《》？/?<>.;:'\"[]{}_)(^$!`"

def split_word():
    with open(r'test_data','rb') as fp:
        content = fp.read()
    try:
        content = content.decode('utf8')
    except:pass
    content = content.upper()
    content = content.replace("x20",'')
    content = content.replace("nbsp",'')
    content = content.replace("xxx",'')
    for word in unused_words:
        content = content.replace(word,'')
    content = content.replace("\r\n",',')
    wordlist = jieba.cut(content)
    
    newcontent = " ".join(wordlist)
    with open(r'splitWord','wb') as fp:
        fp.write(newcontent)
    
if __name__ == "__main__":
    #split_word()
    WordList = [u"赤壁",u"时空"]
    AA = BiGram() 
    AA.init_modle()
    print AA.SentenceProbability(WordList)
    ResultWords = AA.GetMostLikelyWord(u"赤壁")
    for Word in ResultWords:print Word
