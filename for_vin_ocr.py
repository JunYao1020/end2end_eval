import base64
import json

import requests

from file_process import get_image_file_list, write_str_list_to_txt


def compare_vin():
    res_list = []
    image_vin = {}
    vin_ocr_url = "http://192.168.60.30:9809/ocr/prediction"
    with open("20220705_vin_url.txt", encoding='utf-8') as lines:
        for line in lines:
            split_res = line.split(' ')
            if len(split_res) < 2:
                continue
            image_path = split_res[0]
            image_name = image_path[image_path.rfind('/') + 1:]
            vin = split_res[1].strip()
            image_vin[image_name] = vin

    image_list = get_image_file_list("compare_yitu_vin")
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

    write_str_list_to_txt(res_list, "diff_vin.txt")
