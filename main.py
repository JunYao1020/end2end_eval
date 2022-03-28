from cjml_clas import CjmlClas
from cjml_detection import CjmlDetection
from cjml_ocr import *
from evaluation import *
from file_process import *
from rotate_image import *

det_model_dir = 'infer_third_it5/det'
rec_model_dir = 'infer_third_it5/rec'
cls_model_dir = 'infer_third_it5/cls'

rec_char_dict_path = 'dict/ppocr_keys_v1.txt'

all_path = 'eval_first2_it5/res_second_all.txt'
slope_path = 'eval_first2_it5/res_second_slope.txt'


def start_ocr_eval_by_image(image_dir_slope, image_dir_all, label_path):
    model_args = {'det_model_dir': det_model_dir, 'rec_model_dir': rec_model_dir
                  , 'cls_model_dir': cls_model_dir, 'rec_char_dict_path': rec_char_dict_path}

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
    ocr = CjmlOcr(model_args)
    res_slope = ocr.ocr(image_dir_slope)
    res_slope_str = json.dumps(res_slope)
    with open("eval_third_it5/res_second_slope.txt", "w") as f:
        f.write(res_slope_str)

    res_all = ocr.ocr(image_dir_all)
    res_all_str = json.dumps(res_all)
    with open("eval_third_it5/res_second_all.txt", "w") as f:
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


def start_ocr_eval_by_predict_label(predict_slope_path, predict_all_path, label_path):
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

    res_all = txt_dict2dict(predict_all_path)
    res_slope = txt_dict2dict(predict_slope_path)
    process_real_dict(standard_label)
    res_slope_score, list_slope_res = eval_ocr(res_slope, standard_label)
    res_all_score, list_all_res = eval_ocr(res_all, standard_label)
    print(res_slope_score, res_all_score)


def generate_det_res(no_clas_dir='D:\\0310_no_clas\\'
                     , no_det_dir='D:\\0310_no_det\\'
                     , det_des_dir='D:\\0310_det\\'
                     , all_det_des_dir='D:\\0310_all_det\\'
                     , no_right_det_dir='D:\\no_right_det_dir\\'
                     ):
    if not os.path.exists(no_det_dir):
        os.makedirs(no_det_dir)
    if not os.path.exists(det_des_dir):
        os.makedirs(det_des_dir)
    if not os.path.exists(all_det_des_dir):
        os.makedirs(all_det_des_dir)
    if not os.path.exists(no_right_det_dir):
        os.makedirs(no_right_det_dir)

    image_path_list = get_image_file_list(no_clas_dir)
    det = CjmlDetection()
    clas = CjmlClas()

    for i in image_path_list:
        if i[-5: -4] == '本':
            continue
        class_id = clas.get_image_class(i)
        angle = class_id % 1000
        rotated_img, rotated_img_path = rotate_image(i, no_det_dir, angle)

        if angle % 10 != 0:
            location = det.get_location(rotated_img_path)
            location = [int(f) for f in location]
            det_img = rotated_img[location[1]: location[1] + location[3], location[0]: location[0] + location[2]]
            cv2.imwrite(det_des_dir + i[i.rfind('\\') + 1:], det_img)
            cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_img)
        else:
            rotated_right_img, rotated_right_img_path = rotate_right_image(rotated_img_path, no_right_det_dir)
            location = det.get_location(rotated_right_img_path)
            location = [int(f) for f in location]
            det_right_img = rotated_right_img[location[1]: location[1] + location[3],
                                              location[0]: location[0] + location[2]]
            cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_right_img)

            cv2.imwrite(det_des_dir + i[i.rfind('\\') + 1:], rotated_img)


def generate_det_res_new(no_clas_dir='D:\\0310_no_clas\\'
                         , no_det_dir='D:\\0310_no_det\\'
                         , det_des_dir='D:\\0310_det\\'
                         , all_det_des_dir='D:\\0310_all_det\\'
                         , no_right_det_des_dir='D:\\0310_no_right_det\\'):
    if not os.path.exists(no_det_dir):
        os.makedirs(no_det_dir)
    if not os.path.exists(det_des_dir):
        os.makedirs(det_des_dir)
    if not os.path.exists(all_det_des_dir):
        os.makedirs(all_det_des_dir)
    if not os.path.exists(no_right_det_des_dir):
        os.makedirs(no_right_det_des_dir)

    image_path_list = get_image_file_list(no_clas_dir)
    det = CjmlDetection()
    clas = CjmlClas()

    for i in image_path_list:
        if i[-5: -4] == '本':
            continue
        class_id = clas.get_image_class(i)
        angle = class_id % 1000
        rotated_img, rotated_img_path = rotate_image(i, no_det_dir, angle)

        if angle % 10 != 0:
            location = det.get_location(rotated_img_path)
            location = [int(f) for f in location]
            det_img = rotated_img[location[1]: location[1] + location[3], location[0]: location[0] + location[2]]
            cv2.imwrite(det_des_dir + i[i.rfind('\\') + 1:], det_img)
            cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_img)
        else:
            rotated_right_img, rotated_right_img_path = rotate_right_image(rotated_img_path,
                                                                           no_right_det_des_dir)
            location = det.get_location(rotated_right_img_path)
            location = [int(f) for f in location]
            det_right_img = rotated_right_img[location[1]: location[1] + location[3],
                            location[0]: location[0] + location[2]]
            cv2.imwrite(all_det_des_dir + i[i.rfind('\\') + 1:], det_right_img)

            cv2.imwrite(det_des_dir + i[i.rfind('\\') + 1:], rotated_img)


if __name__ == '__main__':
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\download_image\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_third_it5\\no_det\\'
    det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_third_it5\\det_des\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_third_it5\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\eval_third_it5\\no_right_det_des\\'
    # generate_det_res(no_clas, no_det, det_des, all_det_des, no_right_det_des)
    start_ocr_eval_by_image(det_des, all_det_des, 'Label.txt')
