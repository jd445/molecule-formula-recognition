from bs4 import BeautifulSoup
import re
import xlwt
nice =[]
global No
def analysis(response,n):
    try:
        bs = BeautifulSoup(response, "html.parser")
        title2 = bs.find_all('meta')
        savedata(str(title2[32]),n)
    except:
        pass
No = 0

import csv
import codecs
def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name,'w+','utf-8')#追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

def savedata(title1,n):
    try:
        global No
        No = No + 1
        num = re.sub(r'CID .*$', "", title1)
        a=num.split('|')[1]
        nice.append([n,a])
        # print(n,a)
    except:
        pass
for i in range(500,45000):
    try:
        with open("{}.txt".format(i), "r",encoding = 'gb2312') as f:  # 打开文件
            data = f.read()  # 读取文件
            analysis(data,i)
            f.close()
    except:
        pass
print(nice)
data_write_csv("1.csv",nice)
