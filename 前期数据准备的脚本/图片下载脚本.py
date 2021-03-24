
import requests
from multiprocessing import Pool


def mission(url, n):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"}
    print('这是啥', n)
    response = requests.get(url, headers=headers)
    print('他来了', n)
    f = open("{}.ts".format(n), "wb")
    f.write(response.content)
    f.close()
    print("成功")


if __name__ == "__main__":
    pool = Pool(30)
    for n in range(0, 500):
        ""
        url = "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=50&width=500&height=500"
        pool.apply_async(mission, (url, n))
    pool.close()
    pool.join()
