import cv2
import imutils
import numpy as np
import requests
from PIL import Image, ImageFont, ImageDraw


def download_img(des_path, url):
    """
    根据url下载图片到指定目录

    :param des_path: 下载保存父目录
    :param url: 图片地址
    :return: 图片本地路径
    """
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
    """
    生成指定大小的空白图片
    :param size: 指定大小
    :return: 空白图片
    """
    return Image.new("RGB", size, (255, 255, 255))


def merge_two_pic(one, two):
    """
    上下堆叠两张图片

    :param one: 堆叠在上的图片
    :param two: 堆叠在下的图片
    :return: 堆叠后的图片
    """
    return np.vstack((one, two))


def mark_on_pic(text_list, pic):
    """
    将文本列表中的文本打印在图片上
    :param text_list: 文本列表
    :param pic: 待打印图片
    :return: 打印完成图片
    """
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
    """
    专门为vin-ocr定制的打印图片函数

    :param text_list: 文本列表，仅包含两个文本
    :param pic: 待打印图片
    :return: 打印完成的图片
    """
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


def rotate_image(image_path, des_dir, angle):
    """
    按角度旋转图片

    :param image_path: 待旋转图片路径
    :param angle: 图片斜向的角度
    :param des_dir: 旋转后图片保存路径
    :return: 旋转后的图片 旋转后图片保存的路径
    """
    image_name = image_path[image_path.rfind('\\') + 1:]
    image = cv2.imread(image_path)
    rotated = imutils.rotate_bound(image, -1 * angle)

    if des_dir != '':
        cv2.imwrite(des_dir + image_name, rotated)
    return rotated, des_dir + image_name


def rotate_right_image(image_path, des_dir, angle=15):
    """
    对正向图片进行补黑边

    :param image_path: 待处理图片路径
    :param angle: 补边角度，决定黑边大小，默认15
    :param des_dir: 处理后的图片保存的地址文件夹
    :return: 补边处理后的图片 处理后图片保存的路径
    """
    image_name = image_path[image_path.rfind('\\') + 1:]
    image = cv2.imread(image_path)

    rotated = imutils.rotate_bound(image, -1 * angle)
    rotated = imutils.rotate_bound(rotated, 1 * angle)

    cv2.imwrite(des_dir + image_name, rotated)
    return rotated, des_dir + image_name


def normalize_img(img):
    img = imutils.rotate_bound(img, -8)
    img = imutils.rotate_bound(img, 8)
    return img


if __name__ == '__main__':
    download_img("parallel_clas/",
                 "http://192.168.88.6:9000/vin-plates/20220623/LGWED2A35DE049807-vin_android_1655940172352_ffb9c75ec4504b3b9119d73f6cb7c028.jpg")
