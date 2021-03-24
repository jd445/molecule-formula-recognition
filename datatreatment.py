import torch
from PIL import Image
import matplotlib.pyplot as plt

# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])

unloader = transforms.ToPILImage()
