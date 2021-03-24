
# coding:utf-8

import random
import time

f = open("jj.txt", "r")
fw = open("test.txt", "w")
fww = open("train.txt", "w")


def test():
    raw_list = f.readlines()
    random.shuffle(raw_list)
    print(raw_list)
    for i in range(5000):
        fw.writelines(raw_list[i])
        raw_list.remove(raw_list[i])
    for i in raw_list:
        fww.writelines(i)


if __name__ == "__main__":
    test()
