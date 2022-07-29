import base64
import json
import re
import shutil
import threading
import time
import urllib
from datetime import datetime, timedelta

import cv2
import requests
import urllib3

from cjml_utils.file_util import get_image_file_list


def es_go(start_time, end_time):
    st = start_time.isoformat(sep='T', timespec='microseconds') + '+08:00'
    et = end_time.isoformat(sep='T', timespec='microseconds') + '+08:00'
    search_url = "http://es-cn-tl32ljolq000ewdl0.public.elasticsearch.aliyuncs.com:9200/vinscanresult_behaviorlog_202206/_search"
    query = {"track_total_hits": True,
             "query": {
                 "bool": {
                     "must": [{
                         "term": {
                             "IsSacn": 0
                         }
                     }
                     ],
                     "filter": {
                         "range": {
                             "CreateTime": {
                                 "gt": st,
                                 "lt": et
                             }
                         }
                     }
                 }
             },
             "from": 10,
             "size": 15,
             "sort": [
                 {
                     "CreateTime": {
                         "order": "desc",
                         "unmapped_type": "date"
                     }
                 }
             ]
             }
    query = json.dumps(query)
    headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
    r = requests.get(search_url, data=query, auth=('wujs', 'Wujs!2022'), verify=False, headers=headers)
    result = json.loads(r.text)
    total = result["hits"]["total"]["value"]
    items = result["hits"]["hits"]
    for item in items:
        source = item["_source"]

        vin = source["ResultVinCode"]

        imageUrl = source["OrignalImage"]

        createdAt = source["CreateTime"]
        print(vin, imageUrl, createdAt)

    print(total)
    print(len(items))


def do_something(txt, seconds):
    with open("123w.txt", 'w') as f:
        f.write(txt)
        print("t1 write 1")
        time.sleep(seconds)
        f.write(txt)
        print("t1 write 2")
        print(txt)


def do_something1(txt, seconds):
    with open("123w.txt", 'w') as f:
        time.sleep(1)
        f.write(txt)
        print("t2 write 1")
        time.sleep(seconds)
        f.write(txt)
        print("t2 write 2")
        print(txt)


def io_go():
    t1 = threading.Thread(target=do_something, args=("123", 2))
    t2 = threading.Thread(target=do_something1, args=("wjs", 2))
    t1.start()
    t2.start()


def ocr_predict(image_path):
    alpha_ocr_service_url = "http://192.168.70.21:9798/ocr/prediction"

    with open(image_path, 'rb') as file:
        image = base64.b64encode(file.read()).decode('utf8')

    data = {"key": ["image"], "value": [image]}

    r = requests.post(url=alpha_ocr_service_url, data=json.dumps(data), timeout=(5, 15))
    jresult = r.json()
    bytes_img = jresult['value'][2]

    data = base64.b64decode(bytes_img.encode('utf8'))

    ocr_service_url = 'http://192.168.70.20:9899/predict'
    file = {"img": ("file_name.jpg", data, "image/jpg")}
    r = requests.post(url=ocr_service_url, files=file, timeout=(5, 15))
    print(r.json())
    print(type(bytes_img))


def a_te():
    today = datetime.today()
    st = datetime(today.year, 7, 9)
    et = st + timedelta(days=1)
    st = st.isoformat(sep='T', timespec='microseconds') + '+08:00'
    et = et.isoformat(sep='T', timespec='microseconds') + '+08:00'
    print(st)
    print(et)


def horizontal_vertical_depart(src_dir, des_dir):
    image_path_list = get_image_file_list(src_dir)
    for i in image_path_list:
        im = cv2.imread(i)
        h, w, _ = im.shape
        if h > w:
            shutil.move(i, des_dir)


def iurl():
    uuu = "http://192.168.100.117:9999/20210415/"
    httpx = urllib3.PoolManager()
    res = httpx.urlopen("GET", uuu)
    # res = requests.get(uuu)

    print(res.data.decode())
    subUrls = re.findall(r'<a.*?href=".*?">(.*?)</a>', res.data.decode())
    print(subUrls)


if __name__ == '__main__':
    # today = datetime.today()
    # start_time = datetime(today.year, today.month, today.day)
    # end_time = start_time + timedelta(days=2)
    # es_go(start_time, end_time)
    # ocr_predict('./16LJ9L099V4ZE3815-vin_android_1651717826574_cd76778db2a448e6b4e3513da50d291c.jpg')
    # a_te()
    # horizontal_vertical_depart(r"D:\wjs\vin_clas_data_set\3000", r"D:\wjs\vin_clas_data_set\3180")
    iurl()
