from cjml_clas import CjmlClas
from cjml_detection import CjmlDetection
from cjml_ocr import *
from evaluation import *
from file_process import *
from rotate_image import *

det_model_dir_1 = 'infer/det'
rec_model_dir_1 = 'infer/rec'
cls_model_dir_1 = 'infer/cls'

det_model_dir_2 = 'infer_1/det'
rec_model_dir_2 = 'infer_1/rec'
cls_model_dir_2 = 'infer_1/cls'
rec_char_dict_path = 'dict/ppocr_keys_v1.txt'


def start_ocr_eval(image_dir_one, image_dir_two, label_path):
    one_args = {'det_model_dir': det_model_dir_1, 'rec_model_dir': rec_model_dir_1
                , 'cls_model_dir': cls_model_dir_1, 'rec_char_dict_path': rec_char_dict_path}
    two_args = {'det_model_dir': det_model_dir_2, 'rec_model_dir': rec_model_dir_2
                , 'cls_model_dir': cls_model_dir_2, 'rec_char_dict_path': rec_char_dict_path}

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

    # ocr_00 = CjmlOcr(one_args)
    # ocr_01 = CjmlOcr(two_args)
    # res_00 = ocr_00.ocr(image_dir_one)
    # res_01 = ocr_01.ocr(image_dir_two)
    #
    # res_00_str = json.dumps(res_00)
    # res_01_str = json.dumps(res_01)
    # with open("res00.txt", "w") as f:
    #     f.write(res_00_str)
    # with open("res02.txt", "w") as f:
    #     f.write(res_01_str)

    res_00 = txt_dict2dict('res00.txt')

    # 02是slopedet数据
    res_01 = txt_dict2dict('res02.txt')

    process_real_dict(standard_label)
    res_old, list_one = eval_ocr(res_00, standard_label)
    res_new, list_two = eval_ocr(res_01, standard_label)

    # print(len(set(list_one) & set(list_two)))
    # for s in set(list_one) - set(list_two):
    #     oldpath = 'no_det\\'
    #     newpath = 'nice_no_det\\'
    #     if os.path.exists(oldpath + s):
    #         shutil.copy(oldpath + s, newpath + s)
    print(res_old, res_new)


def tst_det(image_dir_one, image_dir_two, label_path):
    one_args = {'det_model_dir': det_model_dir_1, 'rec_model_dir': rec_model_dir_1
                , 'cls_model_dir': cls_model_dir_1, 'rec_char_dict_path': rec_char_dict_path}
    two_args = {'det_model_dir': det_model_dir_2, 'rec_model_dir': rec_model_dir_2
                , 'cls_model_dir': cls_model_dir_2, 'rec_char_dict_path': rec_char_dict_path}

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

    ocr_00 = CjmlOcr(one_args)
    ocr_01 = CjmlOcr(one_args)
    res_00 = ocr_00.ocr(image_dir_two)
    res_01 = ocr_01.ocr(image_dir_one)

    res_new = eval_ocr(res_00, res_01)
    print(res_new)


def generate_det_res(no_clas_dir='D:\\0310_no_clas\\'
                     , no_det_dir='D:\\0310_no_det\\'
                     , det_des_dir='D:\\0310_det\\'
                     , all_det_des_dir='D:\\0310_all_det\\'
                     , no_right_det_des_dir='D:\\0310_no_right_det\\'):
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
    label = "D:\\wjs\\processed_data_set\\alpha_eval\\Label.txt"
    image_dir_1 = 'D:\\wjs\\processed_data_set\\alpha_eval\\det_data'
    image_dir_2 = 'D:\\wjs\\processed_data_set\\alpha_eval\\det_data'
    no_clas = 'D:\\wjs\\PycharmProjects\\end2end_eval\\download_image\\'
    no_det = 'D:\\wjs\\PycharmProjects\\end2end_eval\\no_det\\'
    det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\det_des\\'
    all_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\all_det_des\\'
    no_right_det_des = 'D:\\wjs\\PycharmProjects\\end2end_eval\\no_right_det_des\\'
    # generate_det_res(no_clas, no_det, det_des, all_det_des, no_right_det_des)
    start_ocr_eval(no_det, det_des, 'Label.txt')

