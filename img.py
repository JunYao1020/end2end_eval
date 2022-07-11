import cv2
import numpy as np
import requests
from PIL import Image, ImageFont, ImageDraw


def download_img(des_path, url):
    try:
        image_name = url[url.rfind('/') + 1:]
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            open(des_path + image_name, 'wb').write(r.content)
            return des_path + image_name
        else:
            print('图片下载失败')
    except Exception as e:
        print("error ----------> ", e)
        return



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


def mark_on_pic4vin(text_list, pic):
    max_len = 0
    for s in text_list:
        if len(s) > max_len:
            max_len = len(s)

    size = pic.size
    gap_y = size[1] // 4
    min_x = size[0] // 2 // max_len
    start_y = size[1] // 4

    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, min(gap_y, min_x))
    draw = ImageDraw.Draw(pic)
    draw.text((0, start_y), text_list[0], font=font, fill='blue')
    draw.text((0, start_y + gap_y), text_list[1], font=font, fill='red')
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


def separate_img(img):
    width = img.shape[1]
    height = img.shape[0]
    return img[:, 0: width // 2], img[:, width // 2: width]


if __name__ == '__main__':
    # onepic = Image.open('wrong_alldet_res/1FA6P8TH3H5785737-vin_android_1646723537172_a748db4f094b4ec0a46cf7d764f9b206.jpg')
    # shape = onepic.size
    # blank_img = generate_blank_image(shape)
    # res = merge_two_pic(onepic, blank_img)
    # cv2.imwrite('merge_res.jpg', res)
    download_img("parallel_clas/", "http://192.168.88.6:9000/vin-plates/20220623/LGWED2A35DE049807-vin_android_1655940172352_ffb9c75ec4504b3b9119d73f6cb7c028.jpg")





