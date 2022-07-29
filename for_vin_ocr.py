import base64
import json
import os
import shutil

import requests

from cjml_utils.file_util import get_image_file_list, write_str_list_to_txt
from cjml_utils.visual_error import visual_diff_vin


def compare_vin(img_dir, compare_txt, res_txt_path):
    res_list = []
    image_vin = {}
    vin_ocr_url = "http://192.168.70.21:9809/ocr/prediction"
    with open(compare_txt, encoding='utf-8') as lines:
        for line in lines:
            split_res = line.split(' ')
            if len(split_res) < 2:
                continue
            image_path = split_res[0]
            image_name = image_path[image_path.rfind('/') + 1:]
            vin = split_res[1].strip()
            image_vin[image_name] = vin

    image_list = get_image_file_list(img_dir)
    for img in image_list:
        img_name = img[img.rfind('\\') + 1:]
        try:
            with open(img, 'rb') as file:
                image = file.read()
            image = base64.b64encode(image).decode()
            data = {"key": ["image"], "value": [image]}
            r = requests.post(url=vin_ocr_url, data=json.dumps(data), timeout=(5, 15))
            vin = r.json()['value'][0]
            if image_vin[img_name] != vin:
                res_list.append(img + " " + image_vin[img_name] + " " + vin)
        except Exception as e:
            print("error --- > ", e)

    write_str_list_to_txt(res_list, res_txt_path)


def start_vis_diff():
    im_dir = 'D:/vin_eval_1k'
    res_txt = 'diff_eval_1k_trt.txt'
    compare_txt = "20220721_vin_url.txt"
    compare_vin(im_dir, compare_txt, res_txt)
    diff_im_dir = 'D:/diff_eval_1k_trt/'
    visual_diff_vin(txt_path=res_txt, save_dir=diff_im_dir)


def gain_diff_img(img_dir, des_dir, sum_im):
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
    cnt = 0
    image_vin = {}
    vin_ocr_url = "http://192.168.70.21:9809/ocr/prediction"
    with open("20220721_vin_url.txt", encoding='utf-8') as lines:
        for line in lines:
            split_res = line.split(' ')
            if len(split_res) < 2:
                continue
            image_path = split_res[0]
            image_name = image_path[image_path.rfind('/') + 1:]
            vin = split_res[1].strip()
            image_vin[image_name] = vin

    image_list = get_image_file_list(img_dir)
    for img in image_list:
        img_name = img[img.rfind('\\') + 1:]
        if cnt > sum_im:
            return
        try:
            with open(img, 'rb') as file:
                image = file.read()
            image = base64.b64encode(image).decode()
            data = {"key": ["image"], "value": [image]}
            r = requests.post(url=vin_ocr_url, data=json.dumps(data), timeout=(5, 15))
            vin = r.json()['value'][0]
            if image_vin[img_name] == vin:
                shutil.move(img, des_dir)
                cnt += 1
            # else:
            #     os.remove(img)
        except Exception as e:
            print("error --- > ", e)

    print("sum: --------->", cnt)


if __name__ == '__main__':
    start_vis_diff()

    # im_dir = 'D:/vin_eval_1k'
    # desi_path = 'D:/diff_eval_1k/'
    # cnt = 9999999999
    # gain_diff_img(im_dir, desi_path, cnt)
