#coding=utf8
'''
Created on 2016-8-2

@author: xuwei

@summary: 中文分词
算法来源：
    <互联网时代的社会语言学：基于SNS的文本数据挖掘>
    http://www.matrix67.com/blog/archives/5044
'''
'''
对于每一个词有：
    1.最大长度：默认词语最大长度max (2-->max)
    2.出现次数：该词在全文中出现的次数
    3.凝固程度：比如说“的电影”和“电影院”，虽然很多时候“的电影”出现的次数比“电影院”出现的频次要高，但凝固程度却要比他低
    4.信息熵：词语的混乱程度（无序程度）

注意：
    语料库的大小和内容不同，会使 凝固程度和信息熵 值的差别较大，因此需要根据不同的语料库调试出合适的参数
'''
from math import log

def find_candidate_word(contnet,maxlength = 8):
    '''
    @summary: 提取候选词(发现 if data in dict 比 if data in dict.keys() 效率高很多很多倍)
    @param maxlength: 词语的最大长度
    @常见停用词： "的", "了", "在", "是", "我", "有", "就", "不", "一个",  "也", "要", "你", "着", "没有", "看", "自己", "这" 
    '''
    CandidateWord = {}
    length = len(contnet)
    for i in range(length):
        if i % 10000 == 0:
            print "已完成 :" + str(i/float(length)*100) + " %"
        for j in range(1,maxlength+1):
            word = contnet[i:i+j]
            if word.find(u',') >= 0 or word.find(u'\r\n') >= 0 or word.find(u'\r') >= 0 or word.find(u'\t') >= 0\
                or word.find(u'\n') >= 0 or word.find(u' ') >= 0 or  word.find(u'的') >= 0 or  word.find(u'了') >= 0 or  word.find(u'我') >= 0 \
                or word.find(u'一个') >= 0 or word.find(u'着') >= 0 or word.find(u'自己') >= 0 or word.find(u'在') >= 0 or word.find(u'就') >= 0\
                or i + j >= length:
                break
            if word in CandidateWord:
                CandidateWord[word][0] += 1
                CandidateWord[word][1] = float(CandidateWord[word][0])/length
                if contnet[i-1] not in CandidateWord[word][5]:
                    CandidateWord[word][5].append(contnet[i-1]) 
                if contnet[i+j] not in CandidateWord[word][6]:
                    CandidateWord[word][6].append(contnet[i+j])
            else:
                if i == 0:
                    last_char_list = []
                else:
                    last_char_list = [contnet[i-1]]
                if i + j >= length:
                    next_char_list = []
                else:
                    next_char_list = [contnet[i+j]]
                CandidateWord[word] = [1,               #出现次数
                                       float(1)/length, #词组出现频率
                                       word,            #候选词
                                       1,               #词语的凝聚度，默认1
                                       0,               #邻字信息熵.默认0
                                       last_char_list,  #这个词的左邻字列表
                                       next_char_list]  #这个词的右邻字列表
    return CandidateWord

def calculated_agglomeration(CandidateWord):
    '''
    @summary: 计算词语凝聚度
    *******************************************************************************************************
    @原文: 那么我们定义
        “电影院”的凝合程度就是 p(电影院) 与 p(电) · p(影院) 比值和 p(电影院) 与 p(电影) · p(院) 的比值中的较小值，
        “的电影”的凝合程度则是 p(的电影) 分别除以 p(的) · p(电影) 和 p(的电) · p(影) 所得的商的较小值。
    *******************************************************************************************************
    1. 凝聚度计算：分为左凝聚度，右凝聚度,去较小值
    2. 左凝聚度计算：
        * 将词组 Word 的第一个字与后面拆分开来,分别为 Begin，Follow
        * left_p = Word_times / (Begin_times * Follow_times)
    3. 右凝聚度计算：
        * 将词组 Word 的最后一个字与前面拆分开来,分别为 Front，End
        * right_p = Word_times / (Front_times * End_times)
    '''
    for key in CandidateWord:
        if len(key) < 2:
            continue
        left_p = CandidateWord[key][1]/(CandidateWord[key[1]][1]*CandidateWord[key[1:]][1])
        right_p = CandidateWord[key][1]/(CandidateWord[key[-1]][1]*CandidateWord[key[:-1]][1])
             
        if left_p < right_p:
            CandidateWord[key][3] = left_p
        else:
            CandidateWord[key][3] = right_p


def calculated_entropy(CandidateWord):
    '''
    @summary: 计算词组的左邻字信息熵 和 右邻字信息熵, 左右邻字越混乱越可能是一个词语
    *******************************************************************************************************
    @原文： 我们用信息熵来衡量一个文本片段的左邻字集合和右邻字集合有多随机。
    -考虑这么一句话:
    -     “吃葡萄不吐葡萄皮不吃葡萄倒吐葡萄皮”，“葡萄”一词出现了四次，其中左邻字分别为 {吃, 吐, 吃, 吐} ，右邻字分别为 {不, 皮, 倒, 皮} 。
    -     根据公式，“葡萄”一词的左邻字的信息熵为 – (1/2) · log(1/2) – (1/2) · log(1/2) ≈ 0.693 ，
    -     它的右邻字的信息熵则为 – (1/2) · log(1/2) – (1/4) · log(1/4) – (1/4) · log(1/4) ≈ 1.04 。
    -     可见，在这个句子中，“葡萄”一词的右邻字更加丰富一些。
    -     我们不妨就把一个文本片段的自由运用程度定义为它的左邻字信息熵和右邻字信息熵中的较小值。
    *******************************************************************************************************
    '''
    for key in CandidateWord:
#         last_char_list = set(CandidateWord[key][5])
#         next_char_list = set(CandidateWord[key][6])
        
        last_char_list = CandidateWord[key][5]
        next_char_list = CandidateWord[key][6]
        left_entropy = 0
        right_entropy = 0
        left_num = len(last_char_list)
        right_num = len(next_char_list)
        #求左邻字信息熵
        if left_num > 0:
            for i in range(left_num):
                left_entropy -= 1.0/left_num * log(1.0/left_num)
        else:
            left_entropy = 0
        #求右邻字信息熵
        if right_num > 0:
            for i in range(right_num):
                right_entropy -= 1.0/right_num * log(1.0/right_num)
        else:
            right_entropy = 0
        #取信息熵中的较小值
        entropy = min([left_entropy,right_entropy])
        
        CandidateWord[key][4] = entropy
        

def screen_word(CandidateWord,times = 5,length = 2,agglome = 50,entropy = 0.5):
    '''
    @summary: 筛选数据
    @param times: 出现次数
    @param length: 词语最小长度
    @param agglome: 词语凝聚度
    @param entropy: 最小邻字信息熵 
    '''
    QualifiedWord = {}
    for key in CandidateWord:
        wordInfo = CandidateWord[key]
#         if key == u"龙泉酒业":
#             print wordInfo
        if wordInfo[0] >= times and len(key) >= length and wordInfo[3] >= agglome and wordInfo[4] >= entropy:
            QualifiedWord[key] = wordInfo
    return QualifiedWord

def replace_punctuation(contnet):
    '''
    @summary: 替换标点符号
    '''
    contnet = contnet.replace(u"\r\n",u",")
    unused_words=u" \t\r\n，。：；“‘”【】『』〉|=+-——（）*&……%￥#@！~·《》？/?<>.;:'\"[]{}_)(^$!`"
    for char in unused_words:
        contnet = contnet.replace(char,u',')
    return contnet

if __name__ == "__main__":
    with open(r'C:\Users\Administrator\Desktop\linshi\10000000.txt','rb') as fp:
        contnet = fp.read()
    if type(contnet) == str:
        try:
            contnet = contnet.decode('utf-8')
        except:pass
    #contnet = replace_punctuation(contnet)
    print "正在生成筛选词..."
    CandidateWord = find_candidate_word(contnet)
    print "正在计算聚合度..."
    calculated_agglomeration(CandidateWord)
    print "正在计算邻字信息熵..."
    calculated_entropy(CandidateWord)
    print "正在筛选词语..."
    '''公司名称词组凝聚度很低，但信邻近字息熵也较低(比如说：华XXX 就有很多，"华"和后面的词语凝聚度就很低)'''
    QualifiedWord = screen_word(CandidateWord,times = 5,length = 3,agglome = 10,entropy=0.4)
    wordlist = QualifiedWord.keys()
    wordcontent = ",".join(wordlist)
    with open("word.txt",'wb') as fp:
        fp.write(wordcontent)
    
            
