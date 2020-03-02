#1、新手小程序：MNIST分类
# 目的：熟悉Pytorch如何构建神经网络及训练过程  （nn.Module）
# 要求：（不要求数据处理，可以调用封装好的数据集）
# 	能够按需求自己构建网络结构
# 	更改封装数据集
# 	Train过程中各超参作用

# encoding: utf-8
import torch
import torch.nn as nn # neural network, torch里一个关于神经网络的包
import torch.nn.functional as F  # 加载nn中的功能函数
import torch.optim as optim  # 加载优化器有关包
import torch.utils.data as Data
from torchvision import datasets, transforms  # 加载计算机视觉有关包
from torch.autograd import Variable

# 每训练64张图片，更新一次权重参数
BATCH_SIZE = 64

# 加载torchvision包内内置的MNIST数据集 这里涉及到transform:将图片转化成torchtensor（28x28）
# 数据集内图片全部转换成向量形式
train_dataset = datasets.MNIST(root='~/data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='~/data/', train=False, transform=transforms.ToTensor())

# 加载小批次数据，即将MNIST数据集中的data分成每组batch_size的小块，shuffle指定是否随机读取
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 定义网络模型亦即Net 这里定义一个简单的全连接层784->10

class Model(nn.Module):  # 继承了nn.Module函数
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 10)  # 线形层，一个网络层

    def forward(self, X):  # 前向传播，X是整个模型的输入，是784x1的向量组成的矩阵（64x784）x (784x10) = (64x10)
        return F.relu(self.linear1(X))


model = Model()  # 实例化全连接层
loss = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数
optimizer = optim.SGD(model.parameters(), lr=0.1)  # SGD:随机梯度下降，对整个模型参数进行更新，lr是学习率
num_epochs = 5  # 整个数据集一共训练5次

# 以下四个列表是为了可视化（暂未实现）
losses = []
acces = []
eval_losses = []
eval_acces = []

# 开始训练
for echo in range(num_epochs):
    train_loss = 0  # 定义训练损失
    train_acc = 0  # 定义训练准确度
    model.train()  # 将网络转化为训练模式，该方法为继承的nn.Module的方法
    for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader，i是迭代序号，train_loader里的数据结构为（图片，标签）
        X = X.view(-1, 784)  # X:[64,1,28,28] -> [64,784]将X向量展平  ，view是torch里的函数，X是torch.tensor类型的
        # X = Variable(X)  # 包装tensor用于自动求梯度
        # label = Variable(label)
        out = model(X)  # 正向传播  X传给forward函数，torch默认模型输入是forward函数
        lossvalue = loss(out, label)  # 求损失值
        optimizer.zero_grad()  # 优化器梯度归零
        lossvalue.backward()  # 反向转播，刷新梯度值，要求在optimizer.zero_grad()后执行
        optimizer.step()  # 优化器运行一步，注意optimizer搜集的是model的参数，将反向传播计算的梯度变化加到权重参数里去

        # 计算损失，损失求和
        train_loss += float(lossvalue)
        # 计算精确度
        _, pred = out.max(1)  # （values, indices索引）取第二维上的最大值 （out是[64, 10]，1就是在10个数里取最大，有64个这样的数）
        # print(out.max(1))
        num_correct = (pred == label).sum()  # pred [64, 1], label [64, 1]
        acc = int(num_correct) / X.shape[0]  # 准确率，X.shape[0] = 64
        train_acc += acc

    losses.append(train_loss / len(train_loader))  # list里添加元素
    acces.append(train_acc / len(train_loader))  # list里添加元素
    print("echo:" + ' ' + str(echo))
    print("lose:" + ' ' + str(train_loss / len(train_loader)))
    print("accuracy:" + ' ' + str(train_acc / len(train_loader)))
    eval_loss = 0
    eval_acc = 0
    model.eval()  # 模型转化为评估模式
    # 进行测试
    for X, label in test_loader:
        X = X.view(-1, 784)
        # X = Variable(X)
        # label = Variable(label)
        testout = model(X)
        testloss = loss(testout, label)
        eval_loss += float(testloss)

        _, pred = testout.max(1)
        num_correct = (pred == label).sum()
        acc = int(num_correct) / X.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))  # list里添加元素
    eval_acces.append(eval_acc / len(test_loader))
    print("testlose: " + str(eval_loss / len(test_loader)))
    print("testaccuracy:" + str(eval_acc / len(test_loader)) + '\n')