import os
import numpy as np
import darknet as dn

darknet_module_dir = os.path.dirname(os.path.realpath(__file__))

DEF_GPU = 0
# ToDO (davidnet): String parameters should be bytes
DEF_CONF: bytes =   f'{darknet_module_dir}/cfg/yolov4.cfg'.encode('ascii')
# DEF_CONF: bytes =   f'{darknet_module_dir}/cfg/yolov4-slowmemory.cfg'.encode('ascii')
DEF_W: bytes =  f'{darknet_module_dir}/yolov4.weights'.encode('ascii')
DEF_DATA: bytes = f'{darknet_module_dir}/cfg/coco.data'.encode('ascii')

if __name__ == "__main__":
    img_array = np.random.randn(300, 300, 3).astype(np.uint8)
    # filepath = "./data/dog.jpg".encode('ascii')
    dn.set_gpu(DEF_GPU)
    net = dn.load_net(DEF_CONF, DEF_W, 0)
    metadata = dn.load_meta(DEF_DATA)
    #ToDO (davidnet): Take out magical constants
    #import cv2
    #custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    #custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    #custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    # response_object = dn.detect(net, metadata, filepath, thresh=0.9, hier_thresh=0.5, nms=.45)
    response_object = dn.detect_image(net, metadata, img_array, thresh=0.9, hier_thresh=0.5, nms=.45)