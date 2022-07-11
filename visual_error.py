from file_process import *


def visual_diff_vin():
    with open("diff_vin.txt", encoding='utf-8') as lines:
        for line in lines:
            split_res = line.split(' ')
            image_path = split_res[0]
            image_name = image_path[image_path.rfind('\\') + 1:]
            yitu_vin = split_res[1].strip()
            cjml_vin = "none"
            if len(split_res) == 3:
                cjml_vin = split_res[2].strip()

            one_pic = Image.open(image_path)
            blank_img = generate_blank_image(one_pic.size)
            two_pic = mark_on_pic4vin(["译图: " + yitu_vin, "cjml: " + cjml_vin], blank_img)
            pic = merge_two_pic(one_pic, two_pic)
            cv2.imwrite('vis_diff_vin/' + image_name, pic)


def old_vis():
    img_dict = get_dict_by_excel('error.xlsx', 'alldet')
    for k, v in img_dict.items():
        one_pic = Image.open(
            'wrong_alldet_res/' + k)
        blank_img = generate_blank_image(one_pic.size)
        two_pic = mark_on_pic(v, blank_img)
        pic = merge_two_pic(one_pic, two_pic)
        cv2.imwrite('wrong_alldet_compare/' + k, pic)


if __name__ == '__main__':
    visual_diff_vin()


