import requests

#
# class OcrService(WebService):
#     def get_pipeline_response(self, read_op):
#         pre_op = PreOp(name="pre", input_ops=[read_op])
#         imgdet_op = ImgdetOp(name="imgdet", input_ops=[pre_op])
#         det_op = DetOp(name="det", input_ops=[imgdet_op])
#         rec_op = RecOp(name="rec", input_ops=[det_op])
#         return rec_op

# class OcrService(WebService):
#    def get_pipeline_response(self, read_op):
#        pre_op = PreOp(name="pre", input_ops=[read_op])
#        det_op = DetOp(name="det", input_ops=[pre_op])
#        # det_op = DetOp(name="det", input_ops=[read_op])
#        # cls_op = ClsOp(name="cls", input_ops=[det_op])
#        rec_op = RecOp(name="rec", input_ops=[det_op])
#        return rec_op

if __name__ == '__main__':
    # # payload = {'key': '4d912e19-bfa7-4201-8b6c-a07464896e7d'}
    # data = {
    #     "msgtype": "text",
    #     "text": {
    #         "content": "hello world, wjs"
    #     }
    # }
    # headers = {'Content-Type': 'application/json'}
    # r = requests.post("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4d912e19-bfa7-4201-8b6c-a07464896e7d"
    #                   , headers=headers, json=data)
    # print(r.status_code)

    print(min(12, 21))