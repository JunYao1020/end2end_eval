import cv2
import numpy as np
import requests
from PIL import Image, ImageFont, ImageDraw


def download_img(des_path, url):
    image_name = url[url.rfind('/') + 1:]
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        open(des_path + image_name, 'wb').write(r.content)
    else:
        print('图片下载失败')


def generate_blank_image(size):
    return Image.new("RGB", size, (255, 255, 255))


def merge_two_pic(one, two):
    return np.vstack((one, two))


def mark_on_pic(text_list, pic):

    max_len = 0
    for s in text_list:
        if len(s) > max_len:
            max_len = len(s)

    size = pic.size
    gap_x = size[0] // 2
    gap_y = size[1] // (len(text_list) + 5)
    min_x = size[0] // 3 // max_len
    start_y = size[1] // (len(text_list) + 2)

    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, min(gap_y, min_x))
    draw = ImageDraw.Draw(pic)
    for text in text_list:
        draw.text((0, start_y), text[0], font=font, fill='blue')
        draw.text((gap_x, start_y), text[1], font=font, fill='red')
        start_y += gap_y

    return pic


def mark_on_pic_for_struct(text_list, pic):

    max_len = 0

    for s in text_list:
        if len(s) > max_len:
            max_len = len(s)

    size = pic.size
    gap_y = size[1] // (len(text_list) + 5)
    min_x = size[0] // (max_len + 3)
    start_x = size[0] // 4
    start_y = size[1] // (len(text_list) + 2)

    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, min(gap_y, min_x))
    draw = ImageDraw.Draw(pic)
    for text in text_list:
        draw.text((start_x, start_y), text, font=font, fill='red')
        start_y += gap_y

    return pic

if __name__ == '__main__':
    # onepic = Image.open('wrong_alldet_res/1FA6P8TH3H5785737-vin_android_1646723537172_a748db4f094b4ec0a46cf7d764f9b206.jpg')
    # shape = onepic.size
    # blank_img = generate_blank_image(shape)
    # res = merge_two_pic(onepic, blank_img)
    # cv2.imwrite('merge_res.jpg', res)
    pass





