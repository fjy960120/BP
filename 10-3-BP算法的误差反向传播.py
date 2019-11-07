#定义一个计算激活函数导数的函数，
def transfer_derivative(output):
    return output * (1.0 - output)
    #之所以上述语句是这个形式，是因为sigmoad函数求导后的函数为上式，直接将这个求导后的式子译成代码
def backward_propagate_error(network,expercted):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()#建立一个保存误差的空列表
        if i != len(network) -1: #network的真实长度如果是4，那么列表中的最大值为三，因为是从零开始标记的，i如果不是最后一个值，执行下列程序
            for j in range(len(layer)):
                error = 0.0  #设置一个初始的误差值
                for neuron in network[i + 1]:##12-14行是在计算每个神经元的加权误差，然后将计算结果跳到19-21行，计算隐含层神经元的纠偏责任
                    error += (neuron['weights'][j] * neuron['responsbility'])
                    errors.append(error) #将更新后的误差添加到误差列表中
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['responsbility'] = errors[j] * transfer_derivative(neuron['output']) #输出层纠偏的责任

if __name__ == '__main__':
    network =[[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614],'output':0.7473771139195361},
               {'weights': [0.2550690257394217, 0.49543508709194095, 0.4494910647887381],'output':0.733450902916955}],
               [{'weights': [0.651592972722763, 0.7887233511355132, 0.0938595867742349],'output':0.7473771139195361},
               {'weights': [0.02834747652200631, 0.8357651039198697, 0.43276706790505337],'output':0.733450902916955}]]
    expected = [0,1]  #期望值为0或者1
    backward_propagate_error(network,expected)
    for layer in network:
        print(layer)