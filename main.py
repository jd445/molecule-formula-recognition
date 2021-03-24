#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
from data_load_treatment import MyDataset, devide_test_train, default_loader
from module import linearnet


def main():
    # 设置数据集大小

    print(torch.__version__)
    print(torch.cuda.is_available())
    devide_test_train(2000)

    text_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    train_data = MyDataset(txt='train.txt', transform=text_transforms)
    test_data = MyDataset(txt='test.txt', transform=text_transforms)
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=10,
        shuffle=True,
        num_workers=4
    )
# dataset：Dataset类型，从其中加载数据
# batch_size：int，可选。每个batch加载多少样本
# shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
# sampler：Sampler，可选。从数据集中采样样本的方法。
# num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
# collate_fn：callable，可选。
# pin_memory：bool，可选
# drop_last：bool，可选。True表示如果最后剩下不完全的batch, 丢弃。False表示不丢弃。
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=10,
        shuffle=False,
        num_workers=4
    )
    print('num_of_trainData:', len(train_data))
    print('num_of_testData:', len(test_data))
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # net = linearnet().cuda()
    net = linearnet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.005)
    criteon = nn.CrossEntropyLoss().to(device)

    # for epoch in range(epoch_num):
    #     for batch_idx, (data, target) in enumerate(train_loader, 0):
    #         data, target = Variable(data).to(device), Variable(target.long()).to(device)
    #         optimizer.zero_grad()  # 梯度清0
    #         output = model(data)[0]  # 前向传播
    #         loss = criterion(output, target)  # 计算误差
    #         loss.backward()  # 反向传播
    #         optimizer.step()  # 更新参数
    #         if batch_idx % 10 == 0:
    #             print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch, batch_idx * len(data), len(train_loader.dataset),
    #                        100. * batch_idx / len(train_loader), loss.item()))

    # torch.save(model, 'cnn.pkl')


    # 接下来是复制粘贴
    for epoch in range(2):

        for batch_idx, (data, target) in enumerate(train_loader,0):
            data = data.view(-1, 100 * 100).to(device)
            target = target.long().to(device)
            optimizer.zero_grad()
            logits = net(data).to(device)
            loss = criteon(logits, target)


            loss.backward()
            # print(w1.grad.norm(), w2.grad.norm())
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        test_loss = 0
        correct = 0

        for data, target in test_loader:
            data = data.view(-1, 100 * 100)
            logits = net(data)
            test_loss += criteon(logits, target).item()

            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print(epoch, test_loss)
    return 0


if __name__ == '__main__':
    main()
