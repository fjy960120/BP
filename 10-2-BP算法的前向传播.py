from math import exp
###调用math数据库中的exp函数，就是以e为底的幂函数

#计算神经元的激励值，定义一个激励函数
def activate(weights,inputs):
    activation = weights[-1]
    #weights[-1]就是权值中的倒数第一个，通常指的是偏置权值，作为加权和的初始值
    ###之所以让i的最大值是weights个数-1，是因为权值中有一个是偏置权值
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

#神经元的传递函数，此处使用了Sigmoid函数
def transfer(activation):
    return 1.0/(1.0 + exp(-activation))

#计算神经网络的正向传递
def forward_propagate(network,row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation =activate(neuron['weights'],inputs)
            ###下面的语句，是在字典中添加一个新属性key=outputs，并给这个属性赋值=激活函数的输出值经过传递函数之后的值
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs
if __name__ == '__main__':
    #测试正向传播
    network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]},
                {'weights': [0.2550690257394217, 0.49543508709194095, 0.4494910647887381]}],
               [{'weights': [0.651592972722763, 0.7887233511355132, 0.0938595867742349]},
                {'weights': [0.02834747652200631, 0.8357651039198697, 0.43276706790505337]}]]
    #下面的语句辨明这个是一个样本，有两个特征值，代表输入层有两个神经元，特征值分别是1，0，最后一个值为None（表示预期的输出值）
    row = [1,0,None]
    output = forward_propagate(network,row)
    print(output)