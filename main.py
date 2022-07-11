from functools import partial

from paddle4cjml.cjml_clas import CjmlClas
from paddle4cjml.cjml_detection import CjmlDetection
from paddle4cjml.cjml_ocr import *
from evaluation import *
from fetch_data_from_db import parallel_process
from file_process import *
from rotate_image import *
import math

det_model_dir = 'infer_first_1k_it6/det'
rec_model_dir = 'infer_first_1k_it6/rec'
cls_model_dir = 'infer_first_1k_it6/cls'

rec_char_dict_path = 'dict/ppocr_keys_v1.txt'


def fetch_label_text_content(label_path):
    """
    获取文本文件中的标注内容
    :param label_path: 待处理的文本文件
    :return:
    """
    standard_label = {}
    with open(label_path, encoding='utf-8') as lines:
        for line in lines:
            image_name = line[line.find('/') + 1: line.find('\t')]
            content = line[line.find('\t') + 1:].replace('false', 'False')
            content_list = eval(content)
            result = []
            for c in content_list:
                result.append(c['transcription'])
            standard_label[image_name] = result
    return standard_label


def start_ocr_eval_by_image(image_dir_slope, image_dir_all, label_path):
    model_args = {'det_model_dir': det_model_dir, 'rec_model_dir': rec_model_dir
                  , 'cls_model_dir': cls_model_dir, 'rec_char_dict_path': rec_char_dict_path}

    standard_label = fetch_label_text_content(label_path)

    ocr = CjmlOcr(model_args)
    res_slope = ocr.ocr(image_dir_slope)
    res_slope_str = json.dumps(res_slope)
    with open("eval_e1_it6_add_det_2/res_second_slope.txt", "w") as f:
        f.write(res_slope_str)

    res_all = ocr.ocr(image_dir_all)
    res_all_str = json.dumps(res_all)
    with open("eval_e1_it6_add_det_2/res_second_all.txt", "w") as f:
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
    with open("eval_e1_it6_add_det_plus/res_second_all.txt", "w") as f:
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

    pfunc = partial(process_generate_det_res_add_det_plus, ocr, clas, det, no_det_dir, all_det_des_dir, no_right_det_dir, add_det_all_dir, crop_dir)
    parallel_process(pfunc, image_path_list)


def process_generate_det_res_add_det_plus(ocr, clas, det, no_det_dir, all_det_des_dir, no_right_det_dir, add_det_all_dir, crop_dir, i):
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


def is_vertical_and_gain_longest_box(box_list):
    """
    判断这个box_list中的检测框是否是近似竖直的，并且返回长度最长的那个检测框
    :param box_list: 待处理的检测框集合
    :return: 检测框是否垂直，最长的检测框
    """
    max_x = 0
    max_y = 0
    res_x = []
    res_y = []
    all_x = 0
    all_y = 0
    for four_box in box_list:
        x_length = np.mean([math.dist(four_box[0], four_box[1]), math.dist(four_box[2], four_box[3])])
        y_length = np.mean([math.dist(four_box[0], four_box[3]), math.dist(four_box[1], four_box[2])])
        all_x += x_length
        all_y += y_length
        if x_length > max_x:
            max_x = x_length
            res_x = four_box
        if y_length > max_y:
            max_y = y_length
            res_y = four_box

    return all_y > all_x, res_y, res_x


def gain_all_area(box_list):
    """
    获取所有检测框的检测面积
    :param box_list: 所有检测框集合
    :return:
    """
    all_x = 0
    for four_box in box_list:
        x_length = np.mean([math.dist(four_box[0], four_box[1]), math.dist(four_box[2], four_box[3])])
        y_length = np.mean([math.dist(four_box[0], four_box[3]), math.dist(four_box[1], four_box[2])])
        all_x += x_length * y_length
    return all_x


def gain_slope(four_box):
    """
    获取检测框的斜率
    :param four_box: 单个检测框
    :return: 该检测框斜率
    """
    slope_one = (four_box[1][1] - four_box[0][1]) / (four_box[1][0] - four_box[0][0])
    slope_two = (four_box[2][1] - four_box[3][1]) / (four_box[2][0] - four_box[3][0])
    slope = (slope_one + slope_two) / 2
    return slope


def gain_slope_vertical(four_box):
    """
    获取垂直检测框斜率
    :param four_box: 单个检测框
    :return: 该检测框垂直斜率
    """
    slope_one = (four_box[2][1] - four_box[1][1]) / (four_box[2][0] - four_box[1][0] + 0.000001)
    slope_two = (four_box[3][1] - four_box[0][1]) / (four_box[3][0] - four_box[0][0] + 0.000001)
    slope = (slope_one + slope_two) / 2
    return slope


def gain_angle_by_slope(slope):
    """
    根据斜率获取角度
    :param slope: 斜率
    :return: 该斜率对应角度
    """
    return math.atan(slope) * 180 / math.pi


def gain_angle_by_add_det(box_list):
    """
    根据检测框集合获取旋转角度
    :param box_list: 检测框集合
    :return: 是否垂直检测框，旋转角度
    """
    flag, longest_y, longest_x = is_vertical_and_gain_longest_box(box_list)
    if len(longest_y) == 0:
        return False, 0
    slope = gain_slope_vertical(longest_y) if flag else gain_slope(longest_x)
    return flag, gain_angle_by_slope(slope)


def start_eval():
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\download_image\\'
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\new_rotate\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\compare_add_det\\no_det\\'
    det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\compare_add_det\\det_des\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\compare_add_det\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\compare_add_det\\no_right_det_des\\'
    add_det_all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\compare_add_det\\add_det_all_det_des\\'
    add_det_slope_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\compare_add_det\\add_det_slope_det_des\\'
    # generate_det_res(no_clas, no_det, det_des, all_det_des, no_right_det_des)
    generate_det_res_add_det(no_clas, no_det, add_det_all_det_des, add_det_slope_det_des, det_des, all_det_des, no_right_det_des)
    # start_ocr_eval_by_image(det_des, all_det_des, 'Label.txt')
    # start_ocr_eval_by_image(det_des, all_det_des, 'Label.txt')


def start_eval_add_det():
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\download_image\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\test_img\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\second_1k_supplier\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_e1_it6_add_det_plus\\no_det\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_e1_it6_add_det_plus\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_e1_it6_add_det_plus\\no_right_det_des\\'
    add_det_all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_e1_it6_add_det_plus\\add_det_all_det_des\\'
    rotate_supply_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_e1_it6_add_det_plus\\rotate_supply_des\\'
    crop_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_e1_it6_add_det_plus\\crop_des\\'
    # generate_det_res_add_det_plus(no_clas, no_det, add_det_all_det_des, all_det_des, rotate_supply_des, no_right_det_des, crop_des)
    start_ocr_eval_by_image_single(crop_des, 'Label.txt')

def start_eval_add_det_hard():
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\200difficult\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\test_img\\'
    # no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\second_1k_supplier\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\200difficult_res\\no_det\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\200difficult_res\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\200difficult_res\\no_right_det_des\\'
    add_det_all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\200difficult_res\\add_det_all_det_des\\'
    rotate_supply_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\200difficult_res\\rotate_supply_des\\'
    crop_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\200difficult_res\\crop_des\\'
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
    # start_eval_add_det()
    # start_eval_add_det_hard()
    # start_eval_test()
    start_eval_test_err()

    # print(multiprocessing.cpu_count())


