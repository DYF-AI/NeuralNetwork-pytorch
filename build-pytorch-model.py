'''
    训练图像分类器的几个步骤：
    1. 使用torchvision加载并且归一化训练和测试数据集
    2. 定义一个神经网络
    3. 定义一个损失函数
    4. 在训练样本数据上训练网络
    5. 在测试样本上测试网络
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

import cv2
from PIL import Image

# torchvision数据集的输出范围是[0,1]的PILImage
# 将他们转为归一化范围为[-1,1]之间的Tensor

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_c, out_c, kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def layer1(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

    def layer2(self, x):
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        return x

    def layer_fc(self, x):
        shape = x.shape
        #x = x.view(-1, shape[0] * shape[1] * shape[2])
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        # layer1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer_fc(x)
        return x

class TrainNet():
    def __init__(self):
        self.epoch_num = 2

    def makeTransform(self):
        self.trans = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def makeTrainDataset(self):
        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=self.trans)
        return trainset

    def makeTrainLoader(self):
        trainset = self.makeTrainDataset()
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=4)

    def makeTestDataset(self):
        testset = torchvision.datasets.CIFAR10(root='./data',
                                                train=False,
                                                download=True,
                                                transform=self.trans)

    def makeTestLoader(self):
        testset = self.makeTestDataset()
        self.testloader= torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=4)

    def makeClasses(self):
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

    def makeNet(self):
        self.model = Net()

    def makeCriterion(self):
        self.cirterion = nn.CrossEntropyLoss()
    
    def makeOptimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def creat(self):
        self.makeNet()
        self.makeTransform()
        self.makeTrainLoader()
        self.makeTestLoader()
        self.makeCriterion()
        self.makeOptimizer()

    def trian(self):
        self.creat()
        for epoch in range(self.epoch_num):
            train_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.cirterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                if i % 1000 == 999:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, train_loss / 1000))
                    train_loss = 0.0
                    # save model
        torch.save(self.model.state_dict(), 'save.pt')
        print("Finished Training")
            
def train():
    trianNet = TrainNet()
    trianNet.trian()

  
if __name__ == "__main__":
    import fire
    fire.Fire()





