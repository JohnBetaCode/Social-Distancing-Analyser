# =============================================================================
from collections import namedtuple
import tensorflow as tf
import cytoolz as cz
import numpy as np

from .category_index_coco import CATEGORY_INDEX

# =============================================================================
def create_pred(tup, size):
    name, score, box = tup
    name = CATEGORY_INDEX[name]
    if size is not None:
        box = pixel_coordinates(box, size)
    box = tuple(map(int, box))
    # change from ymin, xmin, ymax, xmax to xmin,ymin,xmax,ymax
    box = (box[1], box[0], box[3], box[2])
    return dict(name=name["name"], score=score, box=box, id=name["id"],)

def pixel_coordinates(box, size):
    # SHAPES ARE INVERTED
    ymin, xmin, ymax, xmax = box
    return (
        ymin * (size[0] - 1),
        xmin * (size[1] - 1),
        ymax * (size[0] - 1),
        xmax * (size[1] - 1),
    )

def parse_detections(predictions, min_score=0.5, max_predictions=20, image_shape=None):

    if image_shape is not None:
        normalized_predictions = True
    else:
        normalized_predictions = False

    predictions["num_detections"] = int(predictions["num_detections"][0].tolist())
    predictions["classes"] = (
        predictions["detection_classes"][0].astype(np.uint8).tolist()
    )
    predictions["boxes"] = predictions["detection_boxes"][0].tolist()
    predictions["scores"] = predictions["detection_scores"][0].tolist()
    predictions = zip(
        predictions["classes"], predictions["scores"], predictions["boxes"]
    )
    if normalized_predictions:
        predictions = map(lambda tup: create_pred(tup, image_shape), predictions)
    else:
        predictions = map(lambda tup: create_pred(tup, None), predictions)
    predictions = filter(lambda p: p["score"] > min_score, predictions)
    predictions = sorted(predictions, key=lambda p: p["score"])
    predictions = cz.take(max_predictions, predictions)
    predictions = list(predictions)

    return predictions

# =============================================================================
class TFModel(object):
    def __init__(self, saved_model_path, threshold=0.5, normalized=True,
        max_predictions=20):
        # self.imported = tf.saved_model.load_v2(saved_model_path)
        # self.model = self.imported.signatures["serving_default"]
        self.model = tf.contrib.predictor.from_saved_model(saved_model_path)
        self.normalized = normalized
        self.threshold = threshold
        self.max_predictions = max_predictions

    def predict(self, frame):
        image_size = frame.shape[:2] if self.normalized else None
        frame = frame[None, ...]
        outputs = self.model(dict(inputs=frame))
        outputs = parse_detections(
            outputs, 
            image_shape=image_size, 
            min_score=self.threshold,
            max_predictions=self.max_predictions
        )
        return outputs

# =============================================================================