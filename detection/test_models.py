import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from python_path import PythonPath

with PythonPath("."):
    from detection.yolo import Yolo
    from detection.tf_model import TFModel


confid = 0.5
thresh = 0.5

vid_path = "./videos/video.mp4"

weights_path = "./models/yolov3/yolov3.weights"
config_path = "./models/yolov3/yolov3.cfg"
weights_path = "./models/yolov3/yolov3-tiny.weights"
config_path = "./models/yolov3/yolov3-tiny.cfg"
# model = Yolo(weights_path, config_path, thresh)


model_name = Path(Path(os.getenv("MODEL_URL")).stem).stem
saved_model = os.path.join("./models", model_name, "saved_model")
model2 = TFModel(saved_model)

vs = cv2.VideoCapture(vid_path)

while True:
    (grabbed, frame) = vs.read()
    # frame = frame[..., ::-1].copy()
    if not grabbed:
        break

    # predictions = model.predict(frame)
    init = time.time()
    predictions = model2.predict(frame)
    print(f"Time: {time.time()-init}")

    if predictions is not None:

        for pred in predictions:
            if pred["name"] != "person":
                continue
            x, y, x2, y2 = pred["box"]
            w = x2 - x
            h = y2 - y
            cen = [int(x + w / 2), int(y + h / 2)]
            cv2.circle(frame, tuple(cen), 2, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Social distancing analyser", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vs.release()
