# =============================================================================
import numpy as np
import time
import math
import sys
import cv2
import os

sys.path.append(".")

from pathlib import Path
from detection.tf_model import TFModel

from python_utils.video_player import Player

from python_utils.vision_utils import get_base_predictions
from python_utils.vision_utils import draw_predictions
from python_utils.vision_utils import norm_predictions
from python_utils.vision_utils import create_radar
from python_utils.vision_utils import image_resize
from python_utils.vision_utils import Extrinsic
from python_utils.vision_utils import printlog
from python_utils.vision_utils import dotline

# =============================================================================
class SocialDistacing:
    def __init__(self, extrinsic, safe_tresh=2.0, intrinsic=None, draw_warped=True):

        # NOTE: Intrinsic params no supported yet in extrinsic
        self.extrinsic = Extrinsic(ext_file_path=extrinsic, int_file_path=None)

        self.safe_tresh = safe_tresh

        self.__src_detections = []
        self.__dst_detections = []
        self.__src_img = None
        self.__dst_img = None

        self.draw_warped = draw_warped

        self.radar_img = create_radar(
            size=int(os.getenv("SAFE_DISTANCING_RADAR_SIZE", default=300)),
            div=int(os.getenv("SAFE_DISTANCING_RADAR_DIV", default=30)),
            color=(0, 255, 0),
            img=None,
        )
        self.radar_win_name = "warped_space"

    def analyse(self, img_src=None, detections=None):

        if self.extrinsic.M is None or img_src is None:
            return
        self.__src_img = img_src.copy()

        # ---------------------------------------------------------------------
        # Process detection
        if detections is not None:

            self.__src_detections = detections
            self.__src_detections = get_base_predictions(
                predictions=self.__src_detections
            )

            # Normalize predictions coordinates
            if self.__src_img is not None:
                self.__src_detections = norm_predictions(
                    img_shape=self.__src_img.shape, predictions=self.__src_detections
                )

            # Find bounding boxes bases of detections in warped space
            self.__dst_detections = self.__src_detections.copy()
            for idx, src_detec in enumerate(self.__dst_detections):
                src_detec["box_base"][0] = int(
                    src_detec["box_base"][0] * self.extrinsic.src_width
                )
                src_detec["box_base"][1] = int(
                    src_detec["box_base"][1] * self.extrinsic.src_height
                )
                self.__dst_detections[idx]["box_base"] = self.extrinsic.pt_src_to_dst(
                    src_pt=src_detec["box_base"]
                )
                self.__dst_detections[idx]["box_base_norm"] = (
                    src_detec["box_base"][0] / self.extrinsic.dst_warp_size[0],
                    src_detec["box_base"][1] / self.extrinsic.dst_warp_size[1],
                )

        for idx, dst_detec in enumerate(self.__dst_detections):
            for idx2, aux_dst_detec in enumerate(self.__dst_detections):
                if dst_detec["idx"] != aux_dst_detec["idx"]:
                    x1, y1 = dst_detec["box_base"]
                    x2, y2 = aux_dst_detec["box_base"]

                    dx = x1 - x2
                    dy = y1 - y2
                    dx = (
                        dx * self.extrinsic.ppmx
                        if self.extrinsic.ppmx is not None
                        else dx
                    )
                    dy = (
                        dy * self.extrinsic.ppmy
                        if self.extrinsic.ppmy is not None
                        else dy
                    )
                    d = abs(math.sqrt(dx ** 2 + dy ** 2))

                    if d <= self.safe_tresh:
                        dst_detec["safe"] = False
                        dst_detec["neighbors"].append(idx2)

                        x1, y1 = dst_detec["box_base_src"]
                        x2, y2 = aux_dst_detec["box_base_src"]
                        # cv2.line(
                        #     img=self.__src_img,
                        #     pt1=(int(x1), int(y1)),
                        #     pt2=(int(x2), int(y2)),
                        #     color=(0, 0, 255), thickness=2)
                        dotline(
                            src=self.__src_img,
                            p1=(int(x1), int(y1)),
                            p2=(int(x2), int(y2)),
                            color=(0, 0, 255),
                            thickness=1,
                            Dl=5,
                        )

        # ---------------------------------------------------------------------
        # Process image to show analysis of social distancing
        # Draw predictions on src image
        if self.__src_detections is not None:
            self.__src_img = draw_predictions(
                predictions=self.__src_detections,
                img_src=self.__src_img,
                normalized=True,
            )

        # Process warped image

        self.__dst_img = None
        if img_src is not None:
            # get warped image
            self.__dst_img = self.extrinsic.get_warp_img(img_src=img_src)

            self.draw_analysis()

    def draw_analysis(self):

        radar_img = self.radar_img.copy()
        dst_img = image_resize(image=self.__dst_img.copy(), height=radar_img.shape[0])

        # Draw warped space images
        if self.draw_warped:
            # Draw predictions on dst/warped image
            if self.__dst_detections is not None:
                for dst_pt in self.__dst_detections:
                    dst_img = self.draw_detection(
                        img=dst_img, detection=dst_pt, closest=True
                    )
                    radar_img = self.draw_detection(
                        img=radar_img, detection=dst_pt, closest=True
                    )

            cv2.imshow("{}(radar)".format(self.radar_win_name), radar_img)
            cv2.imshow("{}".format(self.radar_win_name), dst_img)

    def draw_detection(
        self, img, detection, closest=True, radius=3, draw_neighbors=True
    ):

        pred_x = int(detection["box_base_norm"][0] * img.shape[1])
        pred_y = int(detection["box_base_norm"][1] * img.shape[0])
        pred_pt = (pred_x, pred_y)

        if draw_neighbors:
            for idx in detection["neighbors"]:
                ng_x = int(
                    self.__dst_detections[idx]["box_base_norm"][0] * img.shape[1]
                )
                ng_y = int(
                    self.__dst_detections[idx]["box_base_norm"][1] * img.shape[0]
                )
                cv2.line(
                    img=img,
                    pt1=pred_pt,
                    pt2=(ng_x, ng_y),
                    color=(255, 255, 255),
                    thickness=1,
                )

        color = (255, 0, 255) if detection["safe"] else (0, 0, 255)

        cv2.circle(img=img, center=pred_pt, radius=radius, color=color, thickness=-1)

        if not detection["safe"]:
            cv2.circle(
                img=img,
                center=pred_pt,
                radius=radius * 2,
                color=(0, 255, 255),
                thickness=1,
            )

        return img

    @property
    def img(self):
        return self.__src_img


# =============================================================================
def main():

    # -------------------------------------------------------------------------
    # Player variables
    WIN_NAME = "Social_Distancing"
    media_player = Player(
        video_src_file=os.path.join("./media", "data_src.yaml"),
        win_name=WIN_NAME,
        media_loop=True,
    )

    # -------------------------------------------------------------------------
    if os.getenv("MODEL_URL") is None:
        printlog(msg="Run first: source configs/env.sh", msg_type="ERROR")
        exit(2)

    # Load object detector model
    model_name = Path(Path(os.getenv("MODEL_URL")).stem).stem
    saved_model = os.path.join("./models", model_name, "saved_model")
    ObjModel = TFModel(
        max_predictions=int(os.getenv("MODEL_MAX_PREDICTIONS", default=20)),
        threshold=float(os.getenv("MODEL_THRESHOLD", default=0.5)),
        saved_model_path=saved_model,
        normalized=True,
    )
    ObjModel.predict(frame=np.zeros((10, 10, 3), dtype=np.uint8))

    # -------------------------------------------------------------------------
    distance_analyser = None
    extrinsic_file = ""
    while True:

        if not media_player._win_pause:

            tick = time.time()

            if media_player.file_extrinsic != extrinsic_file:
                extrinsic_file = media_player.file_extrinsic
                # Load extrinsic from dictionary
                distance_analyser = SocialDistacing(
                    extrinsic=os.path.join("./configs", extrinsic_file),
                    safe_tresh=float(
                        os.getenv("SAFE_DISTANCING_TRESHOLD", default=2.0)
                    ),
                )

            # Run social distancing analyser method
            if distance_analyser.extrinsic.M is not None:

                # Run object detection model and get predictions
                predictions = ObjModel.predict(frame=media_player.cap_img)

                distance_analyser.analyse(
                    img_src=media_player.cap_img, detections=predictions
                )

                # Reproduce result of social distancing analyser
                tock = time.time() - tick
                media_player.reproduce(img_src=distance_analyser.img, process_time=tock)
            else:
                tock = time.time() - tick
                media_player.reproduce(process_time=tock)

        else:
            media_player.reproduce()


# =============================================================================
if __name__ == "__main__":
    main()

# =============================================================================
