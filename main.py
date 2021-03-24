#!/usr/bin/python
# -*- coding: UTF-8 -*-

from PIL import Image
import numpy as np
import torch


def main():

    return 0


if __name__ == '__main__':
    pil_img = Image.open("F:\化学人的大创\开始github\pic_of_molecules\\500.png")
    img = np.array(pil_img)

    print(img)
    main()
