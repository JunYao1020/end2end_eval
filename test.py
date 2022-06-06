import time

import requests
import json
from datetime import datetime, timedelta
import threading


def es_go():
    today = datetime.today()
    start_time = datetime(today.year, today.month, today.day)
    start_time = start_time - timedelta(days=5)
    # start_time = start_time + timedelta(minutes=5)
    end_time = start_time + timedelta(days=1)
    # end_time = start_time + timedelta(seconds=10)
    st = start_time.isoformat(sep='T', timespec='microseconds') + '+08:00'
    et = end_time.isoformat(sep='T', timespec='microseconds') + '+08:00'
    print(st)
    print(et)
    search_url = "http://es-cn-tl32ljolq000ewdl0.public.elasticsearch.aliyuncs.com:9200/vinscanresult_behaviorlog_202205/_search"
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


if __name__ == '__main__':
    es_go()
