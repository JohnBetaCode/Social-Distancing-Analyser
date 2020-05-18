# =============================================================================
import time
import cv2
import sys 
import os

from pathlib import Path

sys.path.append(".") 

from detection.tf_model import TFModel
from python_utils.video_player import Player 

from python_utils.vision_utils import get_base_predictions
from python_utils.vision_utils import draw_predictions
from python_utils.vision_utils import norm_predictions
from python_utils.vision_utils import printlog

# =============================================================================
def main(argv):
    
    # -------------------------------------------------------------------------
    # Player variables
    WIN_NAME = "object_detection_test"
    SRC_FILE = "data_src.yaml"

    fwd = os.path.dirname(os.path.abspath(__file__))
    fwd = os.path.abspath(os.path.join(fwd, os.pardir))
    SRC_PATH = os.path.join(fwd, "media", SRC_FILE)

    media_player = Player(
        video_src_file=SRC_PATH,
        win_name=WIN_NAME,
        media_loop=True)
    
    # -------------------------------------------------------------------------
    if os.getenv("MODEL_URL") is None:
        printlog(msg="Run source configs/env.sh first", msg_type="ERROR")
        exit(2)

    # Load object detector model
    model_name = Path(Path(os.getenv("MODEL_URL")).stem).stem
    saved_model = os.path.join("./models", model_name, "saved_model")
    model2 = TFModel(
        max_predictions=int(os.getenv("MODEL_MAX_PREDICTIONS", default=20)),
        threshold=float(os.getenv("MODEL_THRESHOLD", default=0.5)),
        saved_model_path=saved_model, 
        normalized=True)

    # -------------------------------------------------------------------------
    while True:

        if not media_player._win_pause:

            tick = time.time()

            predictions = model2.predict(frame=media_player.cap_img)
            
            norm_preds = norm_predictions(
                img_shape=media_player.cap_img.shape,
                predictions=predictions)
            get_base_predictions(predictions=norm_preds)

            img = draw_predictions(
                img_src=media_player.cap_img, 
                predictions=norm_preds,
                normalized=True)

            tock = time.time() - tick

            media_player.reproduce(
                img_src=img, process_time=tock)

        else:
            media_player.reproduce()

# =============================================================================
if __name__ == "__main__":
    main(sys.argv[1:])

# =============================================================================