from file_process import *

if __name__ == '__main__':
    img_dict = get_dict_by_excel('error.xlsx', 'alldet')
    for k, v in img_dict.items():
        one_pic = Image.open(
            'wrong_alldet_res/' + k)
        blank_img = generate_blank_image(one_pic.size)
        two_pic = mark_on_pic(v, blank_img)
        pic = merge_two_pic(one_pic, two_pic)
        cv2.imwrite('wrong_alldet_compare/' + k, pic)

