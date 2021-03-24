# -*- coding=utf-8 -*-
from bs4 import BeautifulSoup
import urllib.request
import urllib.error
import re
import sqlite3
import requests
import xlwt
import html
headers = {
    'User-Agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36 Edg/85.0.564.51'
}
nice =[]
global No
No = 0
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    i = 0
    for data in datas:
        for j in range(len(data)):
            sheet1.write(i, j, data[j])
        i = i + 1

    f.save(file_path)  # 保存文件

def main():
    for n in range(500, 50000):
        url = "https://pubchem.ncbi.nlm.nih.gov/compound/{}".format(n)
        getdata(url)


def getdata(url):
    req = urllib.request.Request(url=url, headers=headers)
    response = urllib.request.urlopen(req)
    # 9 -----html
    htmlok = response.read().decode('utf-8')
    analysis(htmlok, url)


def analysis(response, url):
    bs = BeautifulSoup(response, "html.parser")
    title1 = bs.title.string
    title2 = bs.find_all('meta')
    savedata(str(title2[32]))


def savedata(title1):
    try:
        global No
        No = No + 1
        num = re.sub(r'CID .*$', "", title1)
        a=num.split('|')[1]
        nice.append(a)
        print(No,a)
    except:
        pass


if __name__ == '__main__':
    main()
    data_write("niu.xlsx",nice)
