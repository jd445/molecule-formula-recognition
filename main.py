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


def main():
    # 设置数据集大小
    devide_test_train(2000)
    train_data = MyDataset(txt='train.txt', transform=transforms.ToTensor())
    test_data = MyDataset(txt='test.txt', transform=transforms.ToTensor())
    # 然后就是调用DataLoader和刚刚创建的数据集，来创建dataloader，这里提一句，loader的长度是有多少个batch，所以和batch_size有关
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=6,
        shuffle=True,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=6,
        shuffle=False,
        num_workers=4
    )
    print('num_of_trainData:', len(train_data))
    print('num_of_testData:', len(test_data))
    return 0


if __name__ == '__main__':
    # pil_img = Image.open("F:\化学人的大创\开始github\pic_of_molecules\\500.png")
    # img = np.array(pil_img)
    main()
