#coding=utf8
'''
Created on 2016-7-29

@author: xuwei

@summary: Trie树
Trie树，也被称为字典树，前缀树，是一种以树的形式存储数据的结构。
其思想为，利用字符串公共前缀来降低查询时间的开销以提高检索效率。
'''
'''
数据结构：
hello,her,hi
abcd,abc,abd

Root:   Value:__None
        Child:__a   Value:__None
             |      Child:__b    Value:__None
             |                   Child:__c    Value:__abc
             |                        |       Child:__d    Value:__abcd
             |                        |                    Child:__{}
             |                        |__d    Value:__abd
             |                                Child:__{}
             |
             |__h    Value:__None
                     Child:__e    Value:__None
                          |       Child:__l    Value:__None
                          |            |       Child:__l    Value:__None
                          |            |                    Child:__o    Value:__hello
                          |            |                                 Child:__{}
                          |            |__r    Value:__her
                          |                    Child:__{}
                          |__i    Value:__hi
                                  Child:__{}
'''
import re

class Node(object):
    def __init__(self):
        '''
        @summary: 树节点
        '''                                             
        self.value = None
        self.children = {}                                                                                   

class TrieTree(object):
    '''
    @summary: Trie树结构及操作
    '''
    def __init__(self):
        self.root = Node()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                child = Node()
                node.children[char] = child
                node = child
            else:
                node = node.children[char]
        node.value = word

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            else:
                node = node.children[char]
        return node.value
    
    def check_path(self,word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            else:
                node = node.children[char]
        return True
    
    
    def word_segment_full(self,string,maxlen):
        '''
        @summary: 全匹配
        '''
        word_list = []
        split_list = re.findall(u'''[\u4e00-\u9fa5\w]+''',string)
        if split_list:
            string = " ".join(split_list)
        length = len(string)
        if length < maxlen:
            maxlen = length
        for i in xrange(length):
            for j in range(i+1,maxlen + 1):
                ret_word = ''.join(string[i:j])
                if self.search(ret_word):
                    word_list.append(ret_word)
                    continue
            if string[i] not in word_list[-1]:
                word_list.append(string[i])
        return word_list
                
    def word_segment_length(self,string,maxlen):
        '''
        @summary: 长词优先
        '''
        word_list = []
        split_list = re.findall(u'''[\u4e00-\u9fa5\w]+''',string)
        if split_list:
            string = " ".join(split_list)
        length = len(string)
        if length < maxlen:
            maxlen = length
        for i in xrange(length):
            for j in range(i+1,maxlen + 1):
                ret_word = ''.join(string[i:j])
                if self.search(ret_word):
                    if word_list[-1] in ret_word:
                        word_list.remove(word_list[-1])
                        word_list.append(ret_word)
                    else:
                        word_list.append(ret_word)
                    continue
            if string[i] not in word_list[-1]:
                word_list.append(string[i])
        return word_list
    
    def word_segment_default(self,string):
        '''
        @summary: 默认从左到右匹配
        '''
        word_list = []
        char_list = []
        split_list = re.findall(u'''[\u4e00-\u9fa5\w]+''',string)
        if split_list:
            string = " ".join(split_list)
        for char in string:
            char_list.append(char)
            ret_word = u"".join(char_list)
            if self.check_path(ret_word):
                continue
            if len(char_list) == 1:
                word = char_list[0]
                word_list.append(word)
                char_list = []
            elif len(char_list) > 1:
                word = u"".join(char_list[:-1])
                word_list.append(word)
                char_list = char_list[-1:]
        if char_list:
            word = u"".join(char_list)
            word_list.append(word)
        return word_list
    
    def word_segment(self,string, status = None, maxlen = 10):
        if status == 'full':
            return self.word_segment_full(string, maxlen)
        if status == 'length':
            return self.word_segment_length(string, maxlen)
        else:
            return self.word_segment_default(string)
        
    def display_node(self, node):
        if (node.value != None):
            print node.value
        for char in node.children.keys():
            if char in node.children:
                self.display_node(node.children[char])
        return

    def display(self):
        self.display_node(self.root)

if __name__ == "__main__":
    trie = TrieTree()
    for word in [u'桂林',u'端掉',u'地下',u'兵工厂',u'缴获',u'炮弹',u'地下兵工厂',u'程序员',u'SB',u'大SB',u'种地']:
        trie.insert(word)
    
    string = u"桂林端掉一家“地下兵工厂” 缴获大批炮弹啊啊啊啊 "
    print string
    word_list = trie.word_segment(string)
    print "Word Segment:","/".join(word_list)
    
    print "---------------------------------------------------------"
    
    string = u"程序员都是大SB"
    print string
    word_list = trie.word_segment(string)
    print "Word Segment:","/".join(word_list)
