from random import seed
from random import random
###初始化（initialize）网络
def initialize_network(n_inputs,n_hidden,n_outputs):
    ###n_inputs表示的是输入层的神经元数量，之后的分别是隐藏层和输出层中的神经元
    network = list()

    #初始化隐含层
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]}for i in range(n_hidden)]
    ###上述语句描述的是隐藏层，隐藏层中的权值是用字典表示的，键值对 = 'weights'：value，上述语句中value用一个列表表示
    ###上述语句中键是weight，具体的权值为是一个在输入层神经元+1中的随机数，然而这个键值对的个数与隐藏层神经元的个数相同
    #将隐含层添加到网络中
    network.append(hidden_layer)

    #初始化输出层
    output_layer = [{'weights':[random()for i in range (n_hidden + 1)]} for i in range(n_outputs)]
    #将输出层添加到网络中
    network.append(output_layer)
    return network

if __name__ == '__main__':
    seed(1)
    network = initialize_network(2,2,2)
    for layer in network:
        print(layer)