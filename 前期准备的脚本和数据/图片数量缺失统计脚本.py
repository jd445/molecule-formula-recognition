import os
import sys
from past.builtins import raw_input

d=[str(i) for i in range(500,45000)]
print("输入1 统计班级成员文件缺交情况\n输入2 粘贴或键盘输入统计缺勤情况\n输入3 进行文件同统一改名处理")
p=input()
if p == "1":
    def readname():
        filePath = os.getcwd()
        name = os.listdir(filePath)
        return name
    if __name__ == "__main__":
        name = readname()
        for i in name:
            for j in d:
                if j in i.split('.')[0]:
                    d.remove(j)
    print("让我瞧瞧哪个小可爱没交")
    for i in d:
        print (i)
    input()