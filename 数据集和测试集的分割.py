
# coding:utf-8

import random
import time




def devide_test_train():
    f = open("jj.txt", "r")
    fw = open("test.txt", "w")
    fww = open("train.txt", "w")
    raw_list = f.readlines()
    random.shuffle(raw_list)
    for i in range(2000):
        fw.writelines(raw_list[i])
        raw_list.remove(raw_list[i])
    for i in raw_list:
        fww.writelines(i)


if __name__ == "__main__":
    test()
