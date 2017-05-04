#coding=utf8
'''
Created on 2016-9-6

@author: xuwei

@summary: BP神经网络(三层结构)

注意：在测试遇到了一个问题，我在测试的时候，发现有组测试数据的输出一直不准；
    输出值一个是0.53左右，一个是0.03左右，修改迭代次数也不能解决，对着书上公式看了一天，也没发现什么问题，后来才知道与测试的标签值有关。
    如果训练样本和测试数据中有属性和结果的对应规律较其他样本的规律差别较大，就会导致这种误差（测试样本误差）
    
在人大论坛上看到句话：神经网络的训练样本质量及数量和网络结果严重影响预测结果。
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
    def __init__(self,inputNum,hideNum,outputNum):
        self.inputNum = inputNum + 1
        self.hideNum = hideNum
        self.outputNum = outputNum
        self.init_net_work()
    
    def init_net_work(self):
        '''
        @summary: 初始化三层网络
        '''
        '''输入层的权重为输入层--->隐含层的权重，隐含层的权重为隐含层--->输出层的权重'''
        self.InputLayer = [Neuron(self.hideNum) for _ in range(self.inputNum)]
        self.HideLayer = [Neuron(self.outputNum) for _ in range(self.hideNum)]
        self.OutputLayer = [Neuron() for _ in range(self.outputNum)]
    
    def Forward(self,inputs):
        '''
        @summary: 前向传输
        @输入层不进行计算
        @隐藏层：∑Wij*Oi  Wij:输入层第i单元到隐藏层第j单元的权重；Oi：输入层第i单元的输出值
        @输出层：∑Wjk*Oj  Wjk:隐藏层第j单元到输出层第k单元的权重；Oi：隐藏层第j单元的输出值
        '''
        '''输入层'''
        for i in range(self.inputNum - 1):
            self.InputLayer[i].Activated = inputs[i]
        '''隐藏层'''    
        for j in range(self.hideNum):
            WeightCount = 0.0
            for i in range(self.inputNum):
                WeightCount += self.InputLayer[i].Activated * self.InputLayer[i].Weight[j]
            self.HideLayer[j].Activated = sigmoid(WeightCount)
        '''输出层'''
        output = []
        for k in range(self.outputNum):
            WeightCount = 0.0
            for j in range(self.hideNum):
                WeightCount += self.HideLayer[j].Activated * self.HideLayer[j].Weight[k]
            self.OutputLayer[k].Activated = sigmoid(WeightCount)
            output.append(self.OutputLayer[k].Activated)
            
        return output
        
    def NewBack(self,targets,NowLearn = 0.5,LastLearn = 0.1):
        '''
        @summary: 逆向反馈
        @隐藏层到输出层误差: Ej = Oj(1 - Oj)(Tj - Oj)
        @输入层到隐藏层误差：Ej = Oj(1 - Oj)* ∑(Ek * Wjk)
        @梯度：Gin = Ek * Oj    Ek是对应上一层第k个单元的误差，Oj是当前层第j个单元的输出值
        @权重：ΔWij = λ1 * Gin + λ2 * Gil;  Wij = Wij + ΔWij    
            :其中λ1当前学习速率，λ2上次学习速率(动量因子)，Gn当前梯度，Gl上次梯度,i当前层第i个节点，j下一层第j个节点
            :计算完权重更新梯度
        '''
        
        '''隐藏层到输出层误差'''
        for k in range(self.outputNum):self.OutputLayer[k].Bais = 0.0
        for k in range(self.outputNum):
            Error = targets[k] - self.OutputLayer[k].Activated
            self.OutputLayer[k].Bais = Error * dsigmoid(self.OutputLayer[k].Activated)
            
        '''更新隐藏层到输出层权重、梯度'''
        for j in range(self.hideNum):
            for k in range(self.outputNum):
                nowGradient = self.OutputLayer[k].Bais * self.HideLayer[j].Activated 
                self.HideLayer[j].Weight[k] += NowLearn * nowGradient + LastLearn * self.HideLayer[j].Gradient[k]
                self.HideLayer[j].Gradient[k] = nowGradient

        '''输入层到隐藏层误差'''
        for j in range(self.hideNum):self.HideLayer[j].Bais = 0.0
        for j in range(self.hideNum):
            Error = 0.0
            for k in range(self.outputNum):
                Error += self.OutputLayer[k].Bais * self.HideLayer[j].Weight[k]
            self.HideLayer[j].Bais = Error * dsigmoid(self.HideLayer[j].Activated)
            
        '''更新输入层到隐含层梯度和误差'''
        for i in range(self.inputNum):
            for j in range(self.hideNum):
                nowGradient = self.HideLayer[j].Bais * self.InputLayer[i].Activated
                self.InputLayer[i].Weight[j] += NowLearn * nowGradient + LastLearn * self.InputLayer[i].Gradient[j]
                self.InputLayer[i].Gradient[j] = nowGradient
                
    def Train(self,TrainingSet,SampleOuts,MaxTimes = 4000):
        '''
        @summary: 训练方法
        '''
        for _ in range(MaxTimes):
            for index,train in enumerate(TrainingSet):
                self.Forward(train)
                output = SampleOuts[index]
                self.NewBack(output)
                
    def print_weight(self):
        '''
        @summary: 打印权重
        '''
        for i in range(self.inputNum):print self.InputLayer[i].Weight,
        print
        for i in range(self.hideNum):print self.HideLayer[i].Weight,
        print
    
    def run(self,inputs,out):
        '''
        @summary: 执行测试
        '''
        print self.Forward(inputs),'---->',out
        
    
if __name__ == "__main__":
    TrainingSet =  [[0,0,0,0], [1,0,0,0],[1,1,0, 0],[0,1,0,0],[0,1,1,1]]
    SampleOuts = [[1],[1],[1],[1],[0]]
    inputNum,hideNum,outputNum = (4,4,1)
    AA = NetWork(inputNum,hideNum,outputNum)
    AA.Train(TrainingSet, SampleOuts)
    AA.print_weight()
    for i in range(len(TrainingSet)):
        AA.run(TrainingSet[i],SampleOuts[i])
