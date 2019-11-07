from math import exp
from random import seed
from random import random

#初始化网络
def initialize_network(n_inputs,n_hidden,n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
#计算神经元的激活值（加权和）
def activate(weights,inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation
#定义激活函数
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

#计算神经网络的正向传播
def forward_propagate(network,row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activiation = activate(neuron['weights'],inputs)
            neuron['output'] = transfer(activiation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

#计算激活函数的导数
def transfer_derivative(output):
    return output * (1 - output)

#反向传播误差信息，并将纠偏责任存储在神经元中
def backward_propagate_error(network,expected):
    #reverse()函数是反转函数，若reverse（4，3，2，1），输出的结果就是1，2，3，4
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:   #就是说i不是输出层时的执行下列语句
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i,i + 1]:
                    error += (neuron['weights'][j] * neuron['responsibility'])
                errors.append(error)
        else:  #计算输出层的误差信息
            for j in range(len(layer)):
                neuron = layer[i]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['responsibility'] = error[j] * transfer_derivative(neuron['output'])

#根据误差，更新网络权重
def update_weights(network,row,l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output']for neuron in network[i,i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] = l_rate * neuron['responsibility'] * inputs[j]
            neuron['weights'] [-1] += l_rate * neuron['responsibility']

#根据指定的周期训练网络
def train_network(network,train,l_rate,n_epoch,n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network,row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])** 2 for i in range(len(expected))])
            backward_propagate_error(network,expected)
            update_weights(network,row,l_rate)
        print('')