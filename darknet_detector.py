import os
import numpy as np
import darknet as dn

darknet_module_dir = os.path.dirname(os.path.realpath(__file__))

DEF_GPU = 0
# ToDO (davidnet): String parameters should be bytes
DEF_CONF: bytes = f"{darknet_module_dir}/cfg/yolov4.cfg".encode("ascii")
# DEF_CONF: bytes =   f'{darknet_module_dir}/cfg/yolov4-slowmemory.cfg'.encode('ascii')
DEF_W: bytes = f"{darknet_module_dir}/yolov4.weights".encode("ascii")
DEF_DATA: bytes = f"{darknet_module_dir}/cfg/coco.data".encode("ascii")
IMG_SIZE: int = 608


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin / IMG_SIZE, ymin / IMG_SIZE, xmax / IMG_SIZE, ymax / IMG_SIZE


class ObjectDetector(object):
    def __init__(
        self,
        config_path: str = str(DEF_CONF, encoding="ascii"),
        weights_path: str = str(DEF_W, encoding="ascii"),
        data_path: str = str(DEF_DATA, encoding="ascii"),
        gpu_enabled: bool = True,
    ):

        self._config_path = config_path.encode("ascii")
        self._weights_path = weights_path.encode("ascii")
        self._data_path = data_path.encode("ascii")
        self._gpu_enabled = 0 if gpu_enabled else 1
        dn.set_gpu(self._gpu_enabled)
        self._net = dn.load_net(self._config_path, self._weights_path, 0)
        self._metadata = dn.load_meta(self._data_path)
        self._darknet_image = dn.make_image(
            dn.network_width(self._net), dn.network_height(self._net), 3
        )

    def predict(self, img_array, thresh=0.5, hier_thresh=0.5, nms=0.45):
        """
        Observations: Assert that the image have the correct shapes.
        """
        dn.copy_image_from_bytes(self._darknet_image, img_array.tobytes())

        detections = dn.detect_image(
            self._net, self._metadata, self._darknet_image, thresh, hier_thresh, nms
        )

        predictions = []
        for detection in detections:
            x, y, w, h = detection[2]
            xmin, ymin, xmax, ymax = convertBack(x, y, w, h)
            if xmax - xmin > 0.9 or ymax - ymin > 0.9:
                continue
            name = str(detection[0], encoding="ascii")
            if name != "person":
                continue
            predictions.append(
                {
                    "name": name,
                    "confidence": detection[1],
                    "box": (xmin, ymin, xmax, ymax),
                }
            )

        return predictions


if __name__ == "__main__":
    detector = ObjectDetector()

    # ToDO (davidnet): Take out magical constants
    # Example pipeline
    import cv2

    custom_image_bgr = cv2.imread("test.jpg")  # use: detect(,,imagePath,)
    custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(
        custom_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR
    )

    # Prediction
    response_object = detector.predict(
        custom_image, thresh=0.3, hier_thresh=0.5, nms=0.45
    )
