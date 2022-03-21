import ppdet


class CjmlDetection:
    """
    检测模型
    """
    def __init__(self, config='det_infer/beta/beta.yml', weight='det_infer/best_model.pdparams'):
        """
        初始化一个检测模型

        :param config: 检测模型配置文件路径
        :param weight: 检测模型评估模型路径，注意不是推理模型
        """
        self.cfg = ppdet.core.workspace.load_config(config)
        self.trainer = ppdet.engine.Trainer(self.cfg, mode='test')
        self.trainer.load_weights(weight)

    def get_location(self, image_path):
        """
        根据图片路径获取该图片的检测结果

        :param image_path: 待检测的图片路径
        :return: 检测结果的坐标信息
        """
        return self.trainer.predict([image_path])

