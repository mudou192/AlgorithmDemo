#coding=utf8
'''
Created on 2017-5-2

@author: xuwei

@summary: Hidden Markov Model
本来打算做个 HMM + Double Array Trie 的分词，最后发现我没有训练数据，86年的人民日报效果太差
'''
import os
import json

def viterbi(obs, states, start_p, trans_p, emit_p, def_stat = None):
    '''
    @summary: Viterbi algorithm
    @:param obs: observation sequence
    @:param states: hidden state
    @:param start_p: initial probability（hidden state）
    @:param trans_p: transition probability（hidden state）
    @:param emit_p: emission probability 
    @:param def_stat: default states
    '''
    view = [{} for i in range(len(obs))]
    path = {}
    for stat in states:   
        view[0][stat] = start_p[stat] * emit_p[stat].get(obs[0],0)   
        path[stat] = [stat]
    for i in range(1,len(obs)):
        newpath = {}
        for j in states:
            rets = [(view[i-1][k] * trans_p[k].get(j,0) * emit_p[j].get(obs[i],0) ,k) for k in states if view[i-1][k]>0]
            # If this transition state is not found, default start state.
            if rets:
                (prob, state) = max(rets)
            else:
                (prob, state) = (1, def_stat)
            view[i][j] = prob
            newpath[j] = path[state] + [j]
        path = newpath  
    prob, state = max([(view[-1][y], y) for y in states])
    return prob, path[state]

class HMM(object):
    def __init__(self,trainfile = None):
        if trainfile:
            self.trainfile = trainfile
        else:
            self.trainfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'trainCorpus.txt_utf8')
        
        self.prob_start_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'prob_start.json')
        self.prob_trans_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'prob_trans.json')
        self.prob_emits_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'prob_emits.json')
        
        self.states = ['B','M','E','S']
        self.count_state = {'B':0,'M':0,'E':0,'S':0}
        self.count_start = {'B':0,'M':0,'E':0,'S':0}
        self.count_trans = {'B': {'B':0,'M':0,'E':0,'S':0}, 
                           'M': {'B':0,'M':0,'E':0,'S':0},
                           'E': {'B':0,'M':0,'E':0,'S':0}, 
                           'S': {'B':0,'M':0,'E':0,'S':0}}
        self.count_emits = {'B':{},'M':{},'E':{},'S':{}}
        
        self.prob_start = {}
        self.prob_trans = {}
        self.prob_emits = {}
        self._init_prob_data()

    def __repr__(self):
        return '<Tokenizer-HMM dictionary=%r>' % self.dictionary
    
    def _init_prob_data(self):
        '''initialization probability'''
        if not os.path.exists(self.prob_start_file) or \
            not os.path.exists(self.prob_trans_file) or \
            not os.path.exists(self.prob_emits_file):
            self.calculate_prob()
        else:
            self.prob_start = json.load(open(self.prob_start_file,'rb'))
            self.prob_trans = json.load(open(self.prob_trans_file,'rb'))
            self.prob_emits = json.load(open(self.prob_emits_file,'rb'))
        
    def load_info(self):
        '''load dictionary data'''
        with open(self.trainfile,'rb') as fp:
            for lineno,line in enumerate(fp,1):
                try:
                    line = line.strip().decode('utf-8','ignore')
                    if not line:
                        continue
                    yield line, lineno
                except:
                    raise ValueError('Invalid dictionary entry in %s at Line %d: %s'%(self.trainfile,lineno,line))
    
    def _print_progress(self,lineno):
        if lineno % 10000 == 0:
            print('line number:' + str(lineno) + '\r')
    
    def get_state_list(self,word):
        if len(word) == 1:
            return ['S']
        if len(word) >= 2:
            stats = ['B']
            stats.extend(['M']*(len(word) - 2))
            stats.append('E')
            return stats
        return []
    
    def save_prob(self):
        '''persistence data'''
        json.dump(self.prob_start,open(self.prob_start_file,'wb'),ensure_ascii=False)
        json.dump(self.prob_trans,open(self.prob_trans_file,'wb'),ensure_ascii=False)
        json.dump(self.prob_emits,open(self.prob_emits_file,'wb'),ensure_ascii=False)
    
    def calculate_prob(self):
        '''calculate the probability of each attribute'''
        # count state
        for line, lineno in self.load_info():
            self._print_progress(lineno)
            chars = list(line.replace(' ',''))
            words = line.split(' ')
            stats = reduce(lambda x, y: x + y,map(self.get_state_list,words))
            # enumerate() efficiency is lower than range(len()) o(一︿一+)o
            for i,stat in enumerate(stats):
                if i == 0:
                    self.count_start[stat] += 1
                    self.count_state[stat] += 1
                else:
                    self.count_trans[stats[i-1]][stat] += 1
                    self.count_state[stat] += 1
                    if chars[i] in self.count_emits[stat]:
                        self.count_emits[stat][chars[i]] += 1
                    else:
                        self.count_emits[stat][chars[i]] = 0.0
            
        for k in self.count_start:
            # initial probability
            self.prob_start[k] = self.count_start[k] * 1.0/lineno
         
        for k1 in self.count_trans:
            # transition probability
            for k2 in self.count_trans[k1]:
                if not self.prob_trans.get(k1):
                    self.prob_trans[k1] = {}
                self.prob_trans[k1][k2] = self.count_trans[k1][k2] * 1.0/self.count_state[k1]
                
        for k in self.count_emits:
            # emission probability
            for char in self.count_emits[k]:
                if not self.prob_emits.get(k):
                    self.prob_emits[k] = {}
                self.prob_emits[k][char] = self.count_emits[k][char] * 1.0/self.count_state[k]
        # save calculation probability
        self.save_prob()
            
    def cut(self,sentence):
        _prob,pos_list = viterbi(sentence,self.states, self.prob_start, self.prob_trans, self.prob_emits, def_stat = 'S')
        retword = u''
        words = []
        for i in range(len(pos_list)):
            stat = pos_list[i]
            if stat == 'E' or stat == 'S':
                retword += sentence[i]
                words.append(retword)
                retword = u''
            else:
                retword += sentence[i]
        return words
        
    def test(self,str_test):
        words = self.cut(str_test)
        split_str = "/".join(words)
        return split_str
                    
    
if __name__ == "__main__":
    str_test = u"伦敦是全球最大的现货黄金交易中心，也是全世界金库最密集的地区。与美国期货合约交易所电子报价不同，伦敦现货金价格由做市商银行报价达成。"
    A = HMM()
    split_str = A.test(str_test)
    print split_str
