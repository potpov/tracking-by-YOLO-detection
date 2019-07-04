import numpy as np



########
# YOLO

BATCH_SIZE = 1
MIN_CONFIDENCE = 0.5
NMS_TH = 0.4
RESOLUTION = 416
YOLO_WEIGHT_PATH = "./content/yolov3.weights"
YOLO_CONF_PATH = "./content/yolov3.cfg"
YOLO_START = 0
YOLO_DETECT_IMAGE = False

VIDEO_PATH = './videos/sample.mp4'

CLASS_NUMBER = 80
CLASS_NAME_PATH = "content/data/coco.names"

#########
# PRE-PROCESSING

CAMERA_MATRIX = np.array(
    [
        [9.6542512729631096e+02, 0, 320],
        [0, 9.6542512729631096e+02, 240],
        [0, 0, 1]
    ]
)

DIST_K = np.array([-2.3838808410964285e-01, 0, 0, 0])


UP_TRANSF_MATRIX = np.array(
    [
        [4.82328085e-01, -7.76838417e-01, 3.91526562e+02],
        [4.83429804e-01, 2.52774187e+00, 0.00000000e+00],
        [-1.39016303e-04, 4.44444737e-03, 1.00000000e+00]
    ]
)

BILATERAL_D = 9
BILATERAL_SIGMA = 75

#######
# TRAJ & TRACKING
#######

MAX_LEN_BUCKET = 1000
MIN_TRACKING_TH = 100  # minimum distance to be taken into account for a bucket

########
# others

PALETTE_PATH = './content/pallete'
VIDEO_OUT_PATH = './results/result.avi'

#######
# DEBUG

TESTING = True
SAVE_RESULT = True
LIVE_RESULTS = False
