#coding=utf8
'''
Created on 2016-9-7
@author: xuwei
@summary: 多隐藏层BP神经网络
'''
import math,numpy
def sigmoid(x):
    '''
    @summary: sigmoid 函数，1/(1+e^-x)
    '''
    return 1.0 / (1.0 + math.exp(-x))
def dsigmoid(x):
    """
    @summary: dsigmoid 函数，f'(x) = f(x) * (1 - f(x))
    """
    return x * (1 - x)
class Neuron(object):
    def __init__(self,nextlayerNum = 0):
        '''
        @summary:         神经单元
        @param Weight:    权重
        @param Bais:      偏置(误差)
        @param OutPut/Activated: 输出/激活值(这里激活值就是输出值)
        @param Gradient:  梯度
        '''
        self.Weight = numpy.random.uniform(-1,1,nextlayerNum)
        self.Bais = 0.0
        self.OutPut = None
        self.Activated  = 0.0
        self.Gradient = numpy.random.uniform(0.0,0.0,nextlayerNum)
        
class NetWork(object):
    def __init__(self,inputNum,hideNum,outputNum,hideLayerNum):
        self.inputNum = inputNum + 1
        self.hideNum = hideNum
        self.outputNum = outputNum
        self.hideLayerNum = hideLayerNum
        self.init_net_work()
    
    def init_net_work(self):
        '''
        @summary: 初始化神经网络
        '''
        self.InputLayer = [Neuron(self.hideNum) for _ in range(self.inputNum)]
        
        self.HideLayerList = []
        for n in range(self.hideLayerNum):
            if n == self.hideLayerNum - 1:
                self.HideLayer = [Neuron(self.outputNum) for _ in range(self.hideNum)]
            else:
                self.HideLayer = [Neuron(self.hideNum) for _ in range(self.hideNum)]
            self.HideLayerList.append(self.HideLayer)
            
        self.OutputLayer = [Neuron() for _ in range(self.outputNum)]
    
    def Forward(self,inputs):
        '''
        @summary: 前向传输
        '''
        for i in range(self.inputNum - 1):
            self.InputLayer[i].Activated = inputs[i]
        
        for n in range(self.hideLayerNum):
            NowLayer = self.HideLayerList[n]
            if n == 0:
                LastLayer = self.InputLayer
            else:
                LastLayer = self.HideLayerList[n - 1]
            
            for j in range(len(NowLayer)):
                WeightCount = 0.0
                for i in range(len(LastLayer)):
                    WeightCount += LastLayer[i].Activated * LastLayer[i].Weight[j]
                NowLayer[j].Activated = sigmoid(WeightCount)
            
        output = []
        LastLayer = self.HideLayerList[-1]
        for k in range(self.outputNum):
            WeightCount = 0.0
            for j in range(len(LastLayer)):
                WeightCount += LastLayer[j].Activated * LastLayer[j].Weight[k]
            self.OutputLayer[k].Activated = sigmoid(WeightCount)
            output.append(self.OutputLayer[k].Activated)
            
        return output
        
    def NewBack(self,targets,NowLearn = 0.5,LastLearn = 0.1):
        '''
        @summary: 逆向反馈
        '''
        '''隐藏层到输出层误差:Ej = Oj(1 - Oj)(Tj - Oj)'''
        for k in range(self.outputNum):self.OutputLayer[k].Bais = 0.0
        for k in range(self.outputNum):
            Error = targets[k] - self.OutputLayer[k].Activated
            self.OutputLayer[k].Bais = Error * dsigmoid(self.OutputLayer[k].Activated)
            
        '''更新隐藏层到输出层权重、梯度'''
        for n in range(self.hideLayerNum)[::-1]:
            NowLayer = self.HideLayerList[n]
            if n == self.hideLayerNum - 1:
                NextLayer =  self.OutputLayer
            else:
                NextLayer = self.HideLayerList[n + 1]
            '''输入层到隐藏层误差：Ej = Oj(1 - Oj)* ∑(Ek * Wjk)'''
            for j in range(len(NowLayer)):
                for k in range(len(NextLayer)):
                    nowGradient = NextLayer[k].Bais * NowLayer[j].Activated
                    NowLayer[j].Weight[k] += NowLearn * nowGradient + LastLearn * NowLayer[j].Gradient[k]
                    NowLayer[j].Gradient[k] = nowGradient
                    
            for j in range(len(NowLayer)):NowLayer[j].Bais = 0.0
            for j in range(len(NowLayer)):
                Error = 0.0
                for k in range(len(NextLayer)):
                    Error += NextLayer[k].Bais * NowLayer[j].Weight[k]
                NowLayer[j].Bais = Error * dsigmoid(NowLayer[j].Activated)
            
        '''更新输入层到隐含层梯度和误差'''
        NextLayer = self.HideLayerList[0]
        for i in range(self.inputNum):
            for j in range(self.hideNum):
                nowGradient = NextLayer[j].Bais * self.InputLayer[i].Activated
                self.InputLayer[i].Weight[j] += NowLearn * nowGradient + LastLearn * self.InputLayer[i].Gradient[j]
                self.InputLayer[i].Gradient[j] = nowGradient
                
    def Train(self,TrainingSet,SampleOuts,MaxTimes = 2000):
        '''
        @summary: 训练方法
        '''
        for _ in range(MaxTimes):
            for index,train in enumerate(TrainingSet):
                self.Forward(train)
                output = SampleOuts[index]
                self.NewBack(output)
                
    def run(self,inputs,out):
        '''
        @summary: 执行测试
        '''
        print self.Forward(inputs),'---->',out
        
    
if __name__ == "__main__":
    TrainingSet =  [[0,0,0,0], [1,0,0,0],[1,1,0, 0],[0,1,0,0],[0,1,1,1]]
    SampleOuts = [[1],[1],[1],[1],[0]]
    inputNum,hideNum,outputNum = (4,4,1)
    AA = NetWork(inputNum,hideNum,outputNum,2)
    AA.Train(TrainingSet, SampleOuts)
    
    for i in range(len(TrainingSet)):
        AA.run(TrainingSet[i],SampleOuts[i])
