from paddleclas import PaddleClas


class CjmlClas:
    def __init__(self, inference_model_dir='clas_infer', top_k=1):
        """
        初始化一个分类器

        :param inference_model_dir: 推理模型文件夹
        :param top_k: 预测的top数
        """
        self.clas = PaddleClas(inference_model_dir=inference_model_dir, topk=top_k)

    def get_image_class(self, image_path):
        """
        根据图片的路径获取图片的分类结果

        :param image_path: 图片路径
        :return: 图片分类结果id
        """
        if image_path[-5: -4] == '本':
            return
        result = self.clas.predict(image_path)
        return next(result)[0]['class_ids'][0]

