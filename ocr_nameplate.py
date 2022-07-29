import json
import os
from functools import partial

import cv2

from paddle4cjml.cjml_clas import CjmlClas
from paddle4cjml.cjml_detection import CjmlDetection
from paddle4cjml.cjml_ocr import *
from evaluation import *
from fetch_data_from_db import parallel_process
import math

from cjml_utils.box_util import gain_angle_by_add_det
from cjml_utils.file_util import fetch_label_text_content, txt_dict2dict
from cjml_utils.img_util import rotate_image, rotate_right_image

det_model_dir = 'infer_first_1k_it6/det'
rec_model_dir = 'infer_first_1k_it6/rec'
cls_model_dir = 'infer_first_1k_it6/cls'

rec_char_dict_path = 'dict/ppocr_keys_v1.txt'


def start_ocr_eval_by_image(image_dir_slope, image_dir_all, label_path):
    model_args = {'det_model_dir': det_model_dir, 'rec_model_dir': rec_model_dir
        , 'cls_model_dir': cls_model_dir, 'rec_char_dict_path': rec_char_dict_path}

    standard_label = fetch_label_text_content(label_path)

    ocr = CjmlOcr(model_args)
    res_slope = ocr.ocr(image_dir_slope)
    res_slope_str = json.dumps(res_slope)
    with open("pic/eval_e1_it6_add_det_2/res_second_slope.txt", "w") as f:
        f.write(res_slope_str)

    res_all = ocr.ocr(image_dir_all)
    res_all_str = json.dumps(res_all)
    with open("pic/eval_e1_it6_add_det_2/res_second_all.txt", "w") as f:
        f.write(res_all_str)

    process_real_dict(standard_label)
    res_slope_score, list_slope_res = eval_ocr(res_slope, standard_label)
    res_all_score, list_all_res = eval_ocr(res_all, standard_label)

    # print(len(set(list_one) & set(list_two)))
    # for s in set(list_one) - set(list_two):
    #     oldpath = 'no_det\\'
    #     newpath = 'nice_no_det\\'
    #     if os.path.exists(oldpath + s):
    #         shutil.copy(oldpath + s, newpath + s)
    print(res_slope_score, res_all_score)


def start_ocr_eval_by_image_single(image_dir_all, label_path):
    model_args = {'det_model_dir': det_model_dir, 'rec_model_dir': rec_model_dir
        , 'cls_model_dir': cls_model_dir, 'rec_char_dict_path': rec_char_dict_path}

    standard_label = fetch_label_text_content(label_path)

    ocr = CjmlOcr(model_args)

    res_all = ocr.ocr(image_dir_all)
    res_all_str = json.dumps(res_all)
    with open("pic/eval_e1_it6_add_det_plus/res_second_all.txt", "w") as f:
        f.write(res_all_str)

    process_real_dict(standard_label)
    res_all_score, list_all_res = eval_ocr(res_all, standard_label)
    print(res_all_score)


def start_ocr_eval_by_predict_label(predict_slope_path, predict_all_path, label_path):
    standard_label = fetch_label_text_content(label_path)

    res_all = txt_dict2dict(predict_all_path)
    res_slope = txt_dict2dict(predict_slope_path)
    process_real_dict(standard_label)
    res_slope_score, list_slope_res = eval_ocr(res_slope, standard_label)
    res_all_score, list_all_res = eval_ocr(res_all, standard_label)
    print(res_slope_score, res_all_score)


def generate_det_res_add_det(no_clas_dir
                             , no_det_dir
                             , add_det_all_dir
                             , all_det_des_dir
                             , rotate_supply_dir
                             , no_right_det_dir
                             , crop_dir
                             ):
    if not os.path.exists(no_det_dir):
        os.makedirs(no_det_dir)
    if not os.path.exists(rotate_supply_dir):
        os.makedirs(rotate_supply_dir)
    if not os.path.exists(all_det_des_dir):
        os.makedirs(all_det_des_dir)
    if not os.path.exists(no_right_det_dir):
        os.makedirs(no_right_det_dir)
    if not os.path.exists(add_det_all_dir):
        os.makedirs(add_det_all_dir)
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    image_path_list = get_image_file_list(no_clas_dir)
    det = CjmlDetection()
    clas = CjmlClas(top_k=2)

    model_args = {'det_model_dir': det_model_dir, 'rec_model_dir': rec_model_dir
        , 'cls_model_dir': cls_model_dir, 'rec_char_dict_path': rec_char_dict_path}
    ocr = CjmlOcr(model_args)

    for i in image_path_list:
        if i[-5: -4] == '本':
            continue
        class_and_score_res_list = clas.get_image_top2_class_and_score(i)
        top_1_class = class_and_score_res_list[0][0]

        angle = top_1_class % 1000
        rotated_img, rotated_img_path = rotate_image(i, no_det_dir, angle)
        if angle % 10 != 0:
            location, category_id = det.get_location_and_category_id(rotated_img_path)
            location = [int(f) for f in location]
            det_img = rotated_img[location[1]: location[1] + location[3], location[0]: location[0] + location[2]]
            cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_img)
        else:
            rotated_right_img, rotated_right_img_path = rotate_right_image(rotated_img_path, no_right_det_dir)
            location, category_id = det.get_location_and_category_id(rotated_right_img_path)
            location = [int(f) for f in location]
            det_right_img = rotated_right_img[location[1]: location[1] + location[3],
                            location[0]: location[0] + location[2]]
            cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_right_img)

        i = all_det_des_dir + i[i.rfind('\\') + 1:]

        add_det_all_det_res = list(ocr.ocr(i, rec=False).values())[0]

        flag, add_det_all_det_angle = gain_angle_by_add_det(add_det_all_det_res)
        add_det_img, add_det_img_path = rotate_image(i, add_det_all_dir, add_det_all_det_angle)
        new_i = crop_dir + i[i.rfind('\\') + 1:]
        if flag:
            cv2.imwrite(new_i, add_det_img)
            continue
        width = cv2.imread(i).shape[1]
        add_det_height = add_det_img.shape[0]
        crop_len = int(width * math.sin(abs(add_det_all_det_angle) / 180 * math.pi) / 2)
        crop_img = add_det_img[crop_len: add_det_height - crop_len, :]
        cv2.imwrite(new_i, crop_img)


def generate_det_res_add_det_plus(no_clas_dir
                                  , no_det_dir
                                  , add_det_all_dir
                                  , all_det_des_dir
                                  , rotate_supply_dir
                                  , no_right_det_dir
                                  , crop_dir
                                  ):
    if not os.path.exists(no_det_dir):
        os.makedirs(no_det_dir)
    if not os.path.exists(rotate_supply_dir):
        os.makedirs(rotate_supply_dir)
    if not os.path.exists(all_det_des_dir):
        os.makedirs(all_det_des_dir)
    if not os.path.exists(no_right_det_dir):
        os.makedirs(no_right_det_dir)
    if not os.path.exists(add_det_all_dir):
        os.makedirs(add_det_all_dir)
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    image_path_list = get_image_file_list(no_clas_dir)
    det = CjmlDetection()
    clas = CjmlClas(top_k=2)

    model_args = {'det_model_dir': det_model_dir, 'rec_model_dir': rec_model_dir
        , 'cls_model_dir': cls_model_dir, 'rec_char_dict_path': rec_char_dict_path}
    ocr = CjmlOcr(model_args)

    pfunc = partial(process_generate_det_res_add_det_plus, ocr, clas, det, no_det_dir, all_det_des_dir,
                    no_right_det_dir, add_det_all_dir, crop_dir)
    parallel_process(pfunc, image_path_list)


def process_generate_det_res_add_det_plus(ocr, clas, det, no_det_dir, all_det_des_dir, no_right_det_dir,
                                          add_det_all_dir, crop_dir, i):
    if i[-5: -4] == '本':
        return
    class_and_score_res_list = clas.get_image_top2_class_and_score(i)
    top_1_class = class_and_score_res_list[0][0]

    angle = top_1_class % 1000
    rotated_img, rotated_img_path = rotate_image(i, no_det_dir, angle)
    if angle % 10 != 0:
        location, category_id = det.get_location_and_category_id(rotated_img_path)
        location = [int(f) for f in location]
        det_img = rotated_img[location[1]: location[1] + location[3], location[0]: location[0] + location[2]]
        cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_img)
    else:
        rotated_right_img, rotated_right_img_path = rotate_right_image(rotated_img_path, no_right_det_dir)
        location, category_id = det.get_location_and_category_id(rotated_right_img_path)
        location = [int(f) for f in location]
        det_right_img = rotated_right_img[location[1]: location[1] + location[3],
                        location[0]: location[0] + location[2]]
        cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_right_img)

    i = all_det_des_dir + i[i.rfind('\\') + 1:]

    add_det_all_det_res = list(ocr.ocr(i, rec=False).values())[0]

    flag, add_det_all_det_angle = gain_angle_by_add_det(add_det_all_det_res)
    add_det_img, add_det_img_path = rotate_image(i, add_det_all_dir, add_det_all_det_angle)
    new_i = crop_dir + i[i.rfind('\\') + 1:]
    if flag:
        cv2.imwrite(new_i, add_det_img)
        return
    width = cv2.imread(i).shape[1]
    add_det_height = add_det_img.shape[0]
    crop_len = int(width * math.sin(abs(add_det_all_det_angle) / 180 * math.pi) / 2)
    crop_img = add_det_img[crop_len: add_det_height - crop_len, :]
    cv2.imwrite(new_i, crop_img)


def start_eval_add_det():
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\download_image\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\test_img\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\second_1k_supplier\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\eval_e1_it6_add_det_plus\\no_det\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\eval_e1_it6_add_det_plus\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\eval_e1_it6_add_det_plus\\no_right_det_des\\'
    add_det_all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\eval_e1_it6_add_det_plus\\add_det_all_det_des\\'
    rotate_supply_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\eval_e1_it6_add_det_plus\\rotate_supply_des\\'
    crop_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\eval_e1_it6_add_det_plus\\crop_des\\'
    # generate_det_res_add_det_plus(no_clas, no_det, add_det_all_det_des, all_det_des, rotate_supply_des, no_right_det_des, crop_des)
    start_ocr_eval_by_image_single(crop_des, 'Label.txt')


def start_eval_add_det_hard():
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\200difficult\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\test_img\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\second_1k_supplier\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\200difficult_res\\no_det\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\200difficult_res\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\200difficult_res\\no_right_det_des\\'
    add_det_all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\200difficult_res\\add_det_all_det_des\\'
    rotate_supply_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\200difficult_res\\rotate_supply_des\\'
    crop_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\pic\\200difficult_res\\crop_des\\'
    # generate_det_res_add_det_plus(no_clas, no_det, add_det_all_det_des, all_det_des, rotate_supply_des, no_right_det_des, crop_des)
    start_ocr_eval_by_image_single(crop_des, 'Label_hard.txt')


def start_eval_test():
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\parallel_failed\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\test_img\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\second_1k_supplier\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\parallel_failed\\no_det\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\parallel_failed\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\parallel_failed\\no_right_det_des\\'
    add_det_all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\parallel_failed\\add_det_all_det_des\\'
    rotate_supply_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\parallel_failed\\rotate_supply_des\\'
    crop_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\parallel_failed\\crop_des\\'
    generate_det_res_add_det(no_clas, no_det, add_det_all_det_des, all_det_des
                             , rotate_supply_des, no_right_det_des, crop_des)


def start_eval_test_err():
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\temp_err\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\test_img\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\second_1k_supplier\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\temp_err\\no_det\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\temp_err\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\temp_err\\no_right_det_des\\'
    add_det_all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\temp_err\\add_det_all_det_des\\'
    rotate_supply_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\temp_err\\rotate_supply_des\\'
    crop_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\temp_err\\crop_des\\'
    generate_det_res_add_det(no_clas, no_det, add_det_all_det_des, all_det_des
                             , rotate_supply_des, no_right_det_des, crop_des)


if __name__ == '__main__':
    start_eval_add_det()
    # start_eval_add_det_hard()
    # start_eval_test()
    # start_eval_test_err()

    # print(multiprocessing.cpu_count())
