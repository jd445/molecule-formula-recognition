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


class linearnet(nn.Module):

    def __init__(self):
        super(linearnet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(10000, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1000, 200),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x
