# -----------------------------------------------------------------------------
# DO NOT forget group your variables to keep order, format should be:
#   export PACKGE/FUNCTION_VARIABLE-NAME=VALUE  # [type] values: short description
# Example:
#   export GUI_SENSORS_DISTANCE=1 # [int-bolean](1):Enable/(0):Disable - Distance sensors

# -----------------------------------------------------------------------------
# Intrinsic calibration
export MONO_PATTERN_HOZ=6   # [int][corners] number of horizontal corners in the chessboard
export MONO_PATTERN_VER=4   # [int][corners] number of vertical corners in the chessboard

# -----------------------------------------------------------------------------
# Extrinsic calibration
export VISION_CAL_UNWARPED_WIDTH=300    # [int][pix] width of unwarped image for extrinsics
export VISION_CAL_UNWARPED_HEIGHT=300   # [int][pix] height of unwarped image for extrinsics
export WIN_WARP_HOZ_DIV=25 # [int][pix] warped window horizontal division space
export WIN_WARP_VER_DIV=25 # [int][pix] warped window vertical division space

# -----------------------------------------------------------------------------
# Media Player variables
export WIN_WIDTH=640  # [int][pix] maximum window width size
export WIN_HEIGHT=480 # [int][pix] maximum window height size

# -----------------------------------------------------------------------------
# Object detection
export MODEL_URL="http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
export MODEL_THRESHOLD=0.4      # [float][0-1] threshold/confidence to get boxes of detections
export MODEL_MAX_PREDICTIONS=40 # [int] maximun number of predections of bounding boxes to get from model

# -----------------------------------------------------------------------------
# Social distancing analyser
export SAFE_DISTANCING_TRESHOLD=2.0     # [float][meters] safe distance
export SAFE_DISTANCING_RADAR_SIZE=360   # [int] radar size
export SAFE_DISTANCING_RADAR_DIV=15     # [int] radar vertical and horizontal divitions

# -----------------------------------------------------------------------------