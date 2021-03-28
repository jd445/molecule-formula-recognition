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
from module import ResNet18, ResBlk
import os


def main():
    # 设置数据集大小

    print(torch.__version__)
    print(torch.cuda.is_available())
    devide_test_train(1000)

    text_transforms = transforms.Compose([
        transforms.Resize(100, 100),
        transforms.ToTensor(),
    ])
    train_data = MyDataset(txt='train.txt', transform=text_transforms)
    test_data = MyDataset(txt='test.txt', transform=text_transforms)
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=16,
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
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    print('num_of_trainData:', len(train_data))
    print('num_of_testData:', len(test_data))


#     #########################################################################################################################
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # net = linearnet().cuda()
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    net = linearnet().to(device)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criteon = nn.CrossEntropyLoss().to(device)
    # 接下来是复制粘贴
    for epoch in range(10):

        for batch_idx, (data, target) in enumerate(train_loader, 0):
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
# #########################################################################################################################3

    # device = torch.device('cpu')
    # # model = Lenet5().to(device)
    # model = ResNet18().to(device)

    # criteon = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # print(model)

    # for epoch in range(1000):

    #     model.train()
    #     for batchidx, (x, label) in enumerate(train_loader,0):
    #         print(batchidx)
    #         # [b, 3, 32, 32]
    #         # [b]
    #         x, label = x.to(device), label.to(device)
    #         if hasattr(torch.cuda, 'empty_cache'):
    #             torch.cuda.empty_cache()
    #         logits = model(x)
    #         if hasattr(torch.cuda, 'empty_cache'):
    #             torch.cuda.empty_cache()
    #         # logits: [b, 10]
    #         # label:  [b]
    #         # loss: tensor scalar
    #         loss = criteon(logits, label)

    #         # backprop
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     #
    #     print(epoch, 'loss:', loss.item())

    #     model.eval()
    #     with torch.no_grad():
    #         # test
    #         total_correct = 0
    #         total_num = 0
    #         for x, label in test_loader:
    #             # [b, 3, 32, 32]
    #             # [b]
    #             x, label = x.to(device), label.long().to(device)

    #             # [b, 10]
    #             logits = model(x)
    #             # [b]
    #             pred = logits.argmax(dim=1)
    #             # [b] vs [b] => scalar tensor
    #             correct = torch.eq(pred, label).float().sum().item()
    #             total_correct += correct
    #             total_num += x.size(0)
    #             # print(correct)

    #         acc = total_correct / total_num
    #         print(epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
