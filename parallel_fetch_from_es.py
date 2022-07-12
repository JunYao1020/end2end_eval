import base64
import json
import os
import shutil
from datetime import datetime, timedelta
from functools import partial

import requests

from file_process import write_str_list_to_txt, get_image_file_list
from paddle4cjml.cjml_clas import CjmlClas
from elasticsearch import Elasticsearch
import multiprocessing.dummy as parallel
from img import download_img

cls = {
    1000: 0,
    1090: 0,
    1180: 0,
    1270: 0,
    1315: 0,
    2000: 10000,
    3000: 10000,
    4000: 10000,
    4045: 0,
    4090: 10000,
    4135: 0,
    4180: 10000,
    4225: 0,
    4270: 10000,
    4315: 0,
}
save_clas_dir = "parallel_clas/"
max_size = 10000


def es_go(es_url, es_auth, index, st, et):
    es = Elasticsearch(hosts=es_url, http_auth=es_auth)
    st = st.isoformat(sep='T', timespec='microseconds') + '+08:00'
    et = et.isoformat(sep='T', timespec='microseconds') + '+08:00'
    print(st)
    print(et)
    query = {
        "track_total_hits": True,
        "query": {
            "bool": {
                "must": [{
                    "term": {
                        "IsSacn": 1
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
        "sort": [
            {
                "CreateTime": {
                    "order": "desc",
                    "unmapped_type": "date"
                }
            }
        ]
    }

    # res_count = es.count(index=index, body=count_body, request_timeout=10000)
    # total_size = res_count['count']

    # results = es.search(index=index, body=query, size=100, request_timeout=1000)['hits']['hits']

    page = es.search(index=index, body=query, scroll='5m', size=10000, request_timeout=1000)
    # 第一页结果,为list类型
    results = page['hits']['hits']

    sid = page['_scroll_id']
    scroll_size = page['hits']['total']['value']

    while scroll_size > 0:
        print(len(results))
        page = es.scroll(scroll_id=sid, scroll='2m')
        sid = page['_scroll_id']
        results += page['hits']['hits']
        scroll_size = len(page['hits']['hits'])

    return [x["_source"]["OrignalImage"] for x in results]


def es_go_vin_image_url(es_url, es_auth, index, st, et):
    es = Elasticsearch(hosts=es_url, http_auth=es_auth)
    st = st.isoformat(sep='T', timespec='microseconds') + '+08:00'
    et = et.isoformat(sep='T', timespec='microseconds') + '+08:00'
    print(st)
    print(et)
    query = {
        "track_total_hits": True,
        "query": {
            "bool": {
                "must": [{
                    "term": {
                        "IsSacn": 1
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
        "sort": [
            {
                "CreateTime": {
                    "order": "desc",
                    "unmapped_type": "date"
                }
            }
        ]
    }

    # res_count = es.count(index=index, body=count_body, request_timeout=10000)
    # total_size = res_count['count']

    # results = es.search(index=index, body=query, size=10, request_timeout=1000)['hits']['hits']

    page = es.search(index=index, body=query, scroll='5m', size=10000, request_timeout=1000)
    # 第一页结果,为list类型
    results = page['hits']['hits']

    sid = page['_scroll_id']
    scroll_size = page['hits']['total']['value']

    while scroll_size > 0:
        print(len(results))
        page = es.scroll(scroll_id=sid, scroll='2m')
        sid = page['_scroll_id']
        results += page['hits']['hits']
        scroll_size = len(page['hits']['hits'])

    return [str(x["_source"]["OrignalImage"]) + " " + str(x["_source"]["ResultVinCode"]) for x in results]


def parallel_process(process, path_list):
    pool = parallel.Pool(12)
    pool.map(process, path_list)
    pool.close()
    pool.join()


def process_clas(path):
    try:
        raw_path = download_img(save_clas_dir, path)
        clas = CjmlClas()
        clas_res = clas.get_image_class(raw_path)
    except Exception as e:
        print("error img")
        return
    if cls[clas_res] < max_size:
        save_dir = save_clas_dir + str(clas_res) + "/"
        des_path = save_dir + raw_path[raw_path.rfind('/') + 1:]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # download_img(save_dir, path)
        shutil.move(raw_path, des_path)
        cls[clas_res] += 1
        print("----------->", clas_res, " ", cls[clas_res])
    else:
        try:
            os.remove(raw_path)
        except Exception as e:
            print("remove error")
        sum_r = 0
        for v in cls.values():
            sum_r += v
            if sum_r > 69980:
                print("clas done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


def start_fetch_from_es():
    url = "http://es-cn-tl32ljolq000ewdl0.public.elasticsearch.aliyuncs.com:9200"
    auth = ('wujs', 'Wujs!2022')
    idx = "vinscanresult_behaviorlog_*"
    today = datetime.today()
    start_time = datetime(today.year, 7, 5)
    end_time = start_time + timedelta(days=1)
    return es_go(url, auth, idx, start_time, end_time)


def download_img_from_es():
    urls = start_fetch_from_es()
    # parallel_process(process_clas, urls)
    des_dir = "D:/scan_vin_imgs/"
    pfunc = partial(download_img, des_dir)
    parallel_process(pfunc, urls)


def save_vin_image_url():
    url = "http://es-cn-tl32ljolq000ewdl0.public.elasticsearch.aliyuncs.com:9200"
    auth = ('wujs', 'Wujs!2022')
    idx = "vinscanresult_behaviorlog_*"
    today = datetime.today()
    start_time = datetime(today.year, 7, 5)
    end_time = start_time + timedelta(days=1)
    vin_url = es_go_vin_image_url(url, auth, idx, start_time, end_time)
    write_str_list_to_txt(vin_url, "20220705_vin_url.txt")


if __name__ == '__main__':
    pass

