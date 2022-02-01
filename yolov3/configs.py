#================================================================
#
#   File name   : configs.py
#   Author      : PyLessons
#	Modified 	: Tyler Odenthal - BirdBot
#   Created date: 2020-08-18
#	Modify date : 2021-08-04
#   
#	OG-Website  : https://pylessons.com/
#   MOD-Website : https://www.bird.bot/
#   OG-GitHub   : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   MOD-GitHub  : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#
#	Description : yolov3 configuration file
#
#================================================================

# YOLO options
YOLO_TYPE                   = "yolov4" # yolov4 or yolov3
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_V3_WEIGHTS             = "model_data/yolov3.weights"
YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolov3-tiny.weights"
YOLO_V4_TINY_WEIGHTS        = "model_data/yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = True # 'checkpoints/yolov4-trt-INT8-416' # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.6
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 512
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
if YOLO_TYPE                == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
# Train options
TRAIN_YOLO_TINY             = True
TRAIN_SAVE_BEST_ONLY        = True # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT       = False # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES               = "model_data/Dataset_names.txt"
TRAIN_ANNOT_PATH            = "model_data/Dataset_train.txt"
TRAIN_LOGDIR                = "log"
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}_custom"
TRAIN_LOAD_IMAGES_TO_RAM    = False # With True faster training, but need more RAM
TRAIN_BATCH_SIZE            = 4
TRAIN_INPUT_SIZE            = 512
TRAIN_DATA_AUG              = True
TRAIN_TRANSFER              = True
TRAIN_FROM_CHECKPOINT       = False # "checkpoints/yolov3_custom"
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 2
TRAIN_EPOCHS                = 100

# BirdBot Video Capture Config
WRITE_VIDEO_OUTPUT_FILE     = True
WRITE_VIDEO_GUESS_FILE      = False
WRITE_VIDEO_TRAIN_FILE      = False
HAT_MODE = False

# BirdBot Twitch Bot Config
TWITCH_ACTIVE 				= True
TWITCH_CLIENT_ID 			= ""
TWITCH_NICK 				= ""
TWITCH_TOKEN 				= ""
TWITCH_CHANNELS 			= ""
TWITCH_CLIENT_SECRET 		= ""
MODT = False

# TEST options
TEST_BATCH_SIZE             = 4
TEST_ANNOT_PATH             = "model_data/Dataset_test.txt"
TEST_INPUT_SIZE             = 512
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.4
TEST_IOU_THRESHOLD          = 0.6

if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32]    
    # YOLO_ANCHORS            = [[[23, 27],  [37, 58],   [81,  82]], # this line can be uncommented for default coco weights
    YOLO_ANCHORS            = [[[10, 14],  [23, 27],   [37, 58]],
                               [[81,  82], [135, 169], [344, 319]]]