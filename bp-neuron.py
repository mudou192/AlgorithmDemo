#coding=utf8
'''
Created on 2016-9-1

@author: xuwei

@summary: 
'''
'''
参考：http://blog.jobbole.com/90184/

偏置结点:是为了描述训练数据中没有的特征，偏置结点对于下一层的每一个结点的权重的不同而生产不同的偏置，于是可以认为偏置是每一个结点（除输入层外）的属性。

BP神经网络的训练过程分两部分：

    前向传输，逐层波浪式的传递输出值；
    逆向反馈，反向逐层调整权重和偏置；
    
前向传输（Feed-Forward前向反馈）
    1.在训练网络之前，我们需要随机初始化权重和偏置,对每一个权重取[-1,1]的一个随机实数，每一个偏置取[0,1]的一个随机实数，之后就开始进行前向传输。
    2.为每一个属性设置一个对应的神经元，比如说训练集有100条数据，每条数据有20个属性，那么就设置20个神经元与之对应。
    3.对个属性加权累加，然后加上偏置，最后带入 sigmoid 函数，求得输出值

逆向反馈（Backpropagation）
    我们第一次前向反馈时，整个网络的权重和偏置都是我们随机取，因此网络的输出肯定还不能描述记录的类别，因此需要调整网络的参数，即权重值和偏置值，
    而调整的依据就是网络的输出层的输出值与类别之间的差异，通过调整参数来缩小这个差异，这就是神经网络的优化目标。
    
    

'''

import math,numpy,random

def sigmoid(x):
    '''
    @summary: sigmoid 函数，1/(1+e^-x)
    '''
    return 1.0 / (1.0 + math.exp(-x))

class Neuron(object):
    def __init__(self,nextlayerNum = 1):
        '''
        @summary: 神经元模块
        '''
        '''权重 ∈ [-1,1],为下一层的每一个节点初始化权重'''
        self.Weight = numpy.random.uniform(-1,1,nextlayerNum)
        '''偏置 ∈  {0,1},初始化该节点的偏置
         (有叫作误差，也有叫作阀值：但是感觉阀值这个名称已经偏离了本身的意义，偏置和误差更为合理；阈值不动或者不设置阈值，也是没有问题的，但是有了动态的阈值，那么学习得更快，效果更好) ∈ [0,1]'''
        self.Bais = random.uniform(0,1)
        '''输出参数 ∈ {0,1}'''
        self.OutPut = None
        '''激活值（暂时没有用到）'''
        self.Activated  = None
        
class NeuralNetWork(object):
    def __init__(self,inputNum,hideNum,outputNum,hideDeep):
        '''构建神经网络'''
        '''
        @param inputNum: 输入层个数
        @param hideNum: 隐含层个数
        @param outputNum: 输出层单元个数
        @param hideDeep: 隐含层的层数
        '''
        self.inputNum = inputNum
        self.hideNum = hideNum
        self.outputNum = outputNum
        self.hideDeep = hideDeep
        self.init_net_work()
        
    def init_net_work(self):
        '''
        @summary: 初始化各层神经网络
        '''
        '''初始化输入层'''
        self.InputLayer = [Neuron(self.hideNum) for i in range(self.inputNum)]
        '''初始化隐含层(权重：最后一个是输出层的节点个数，其他是隐藏层结点个数)'''
        self.HideLayers = []
        for i in range(self.hideDeep):
            if i == self.hideDeep - 1:
                HideLayer = [Neuron(self.outputNum) for i in range(self.hideNum)]
            else:
                HideLayer = [Neuron(self.hideNum) for i in range(self.hideNum)]
            self.HideLayers.append(HideLayer)
        '''初始化输出层,输出层没有对下一层的权重，所以可以不使用权重字段'''
        self.OutputLayer =  [Neuron() for i in range(self.outputNum)]
    
    def Forward(self,inputs):
        '''
        @summary: 前向传播
        '''
        '''输入层'''
        for i in range(self.inputNum):
            '''输入层不进行函数处理，只接收输入（将权重修改为1，加权没有影响）'''
            self.InputLayer[i].Weight = self.InputLayer[i].Weight / self.InputLayer[i].Weight
            self.InputLayer[i].OutPut = inputs[i]
            
        '''隐藏层'''
        #NowCount = 0.0
        for i in range(self.hideDeep):
            if i == 0:
                last_layer = self.InputLayer
            else:
                last_layer = self.HideLayers[i-1]
            '''对上一级的数据进行加权计算'''
            for j in range(self.hideNum):
                NowCount = 0.0
                '''1. 对上一层的输出加权进行累加，再加上偏置 '''
                for neural in last_layer:
                    NowCount += neural.OutPut * neural.Weight[j]
                #NowCount = NowCount +  self.HideLayers[i][j].Bais
                '''2. 调用激活函数，得到激活值（即输出值）'''
                self.HideLayers[i][j].OutPut = sigmoid(NowCount)
        
        '''输出层(输出可能是多标签，比如说瓜的品种，和成熟度)'''
        last_layer = self.HideLayers[-1]
        OutTable = []
        for i in range(self.outputNum):
            NowCount = 0.0
            '''1. 对上一层的输出加权然后累加，再加上偏置 '''
            for neural in last_layer:
                NowCount += neural.OutPut * neural.Weight[i]
            #NowCount = NowCount +  self.OutputLayer[i].Bais
            
            '''2. 调用激活函数，得到激活值(激活值即为输出值)'''
            self.OutputLayer[i].OutPut = sigmoid(NowCount)
            OutTable.append(self.OutputLayer[i].OutPut)
            
        return OutTable
    
    def Back(self,learning_rate,output):
        '''
        @summary: 后向传播
        '''
        '''计算输出层误差(梯度)Ej = Oj(1 - Oj)(Tj - Oj)    其中Ej表示第j个结点的误差值，Oj表示第j个结点的输出值，Tj样本输出值'''
        
        for i in range(self.outputNum):
            ERROR = self.OutputLayer[i].OutPut * (1 - self.OutputLayer[i].OutPut) * (output - self.OutputLayer[i].OutPut)
            self.OutputLayer[i].Bais = self.OutputLayer[i].Bais + learning_rate * ERROR
        '''计算隐藏层误差 Ej = Oj(1 - Oj)* ∑(Ek * Wjk) 其中Wjk表示当前层的结点j到下一层的结点k的权重值,Ek下一层的结点k的误差'''
        for i in range(self.hideDeep):
            for j in range(self.hideNum):
                now_layer = self.HideLayers[i]
                if i == self.hideDeep - 1:
                    next_layer = self.OutputLayer
                else:
                    next_layer = self.HideLayers[i + 1]
                ERROR = now_layer[j].OutPut * (1 - now_layer[j].OutPut) * sum([now_layer[j].Weight[k] * next_layer[k].Bais  for k in range(len(next_layer))])
                '''
                @更新权重
                @其中λ表示表示学习速率，取值为0到1，学习速率设置得大，训练收敛更快，但容易陷入局部最优解，学习速率设置得比较小的话，收敛速度较慢，但能一步步逼近全局最优解。
                ΔWij = λEjOi
                Wij = Wij + ΔWij
                @相当于下面使用矩阵的计算
                '''
                now_layer[j].Weight = now_layer[j].Weight * learning_rate * ERROR + now_layer[j].Weight
                '''
                @更新偏置
                Δθj = λEj
                θj = θj + Δθj
                '''
                now_layer[j].Bais = now_layer[j].Bais + learning_rate * ERROR
    
    def print_network(self):
        print "-----------------------------------------------------------------------------"
        for N in self.InputLayer:print N.Weight,
        for i in range(self.hideDeep):
            print "\t"
            for N in self.HideLayers[i]:print N.Weight,
        for N in self.OutputLayer:print N.Weight,
        print 
      
            
    def Train(self,TrainingSet,SampleOuts,learning_rate = 0.1,MaxTimes = 10000):
        '''
        @summary: 训练方法
        '''
        for i in range(MaxTimes):
            for index,train in enumerate(TrainingSet):
                self.Forward(train)
                if i % 100 == 0:
                    self.print_network()
                output = SampleOuts[index]
                if i % 100 == 0:
                    self.print_network()
                self.Back(learning_rate, output)
    
    def run(self,inputs):
        '''
        @summary: 执行测试
        '''
        return self.Forward(inputs)

if __name__ == "__main__":
    TrainingSet =  [[0, 0], [0, 1],[1, 0],[1, 1]]
    SampleOuts = [1,1,1,0]
    inputNum,hideNum,outputNum,hideDeep = (2,2,1,2)
    learning_rate = 0.05
    MaxTimes = 1000
    AA = NeuralNetWork(inputNum,hideNum,outputNum,hideDeep)
    AA.Train(TrainingSet, SampleOuts, learning_rate, MaxTimes)
    print AA.run([0,0])
    print AA.run([0,1])
    print AA.run([1,0])
    print AA.run([1,1])
