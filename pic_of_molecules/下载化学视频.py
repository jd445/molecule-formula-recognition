
import requests
from multiprocessing import Pool


def mission(url, n):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"}
    print('这是啥', n)
    response = requests.get(url, headers=headers)
    print('他来了', n)
    f = open("{}.png".format(n), "wb")
    f.write(response.content)
    f.close()
    print("成功")
#https://pubchem.ncbi.nlm.nih.gov/compound/38184

if __name__ == "__main__":
    pool = Pool(30)
    niubi= [13025,13028,23208,23359,23363,23530,
24351,
24371,
24383,
25487,
26247,
26844,
26849,
26852,
26857,
38184,
39284,
40370,
40501,
42122]
    for n in niubi:
        url = "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid={}&width=500&height=500".format(n)
        pool.apply_async(mission, (url, n))
    pool.close()
    pool.join()
