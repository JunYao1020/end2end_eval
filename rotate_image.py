import cv2
import imutils


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


if __name__ == '__main__':
    rotate_image(r'D:\wjs\clas_data_set\eval_data\4000_error\LN86DDBF5HA046008-vin_ios_1631846872841_16213.jpg'
                 , 'D:\\wjs\\clas_data_set\\eval_data\\4000_error\\temp\\', 45)
