from multiprocessing import Process, Queue, Pipe
import cv2
import os
import time, queue, threading
import random
import numpy as np
import textwrap
# import vlc
import base64
import json
import re
import geocoder
import requests
import subprocess
import tensorflow as tf
import yolov3.twitch_speaker as TCI
import xml.etree.cElementTree as ET
from geopy.geocoders import Nominatim
from threading import Thread
from xml.dom import minidom
from datetime import datetime
from yolov3.configs import *
from yolov3.wallet import *
from yolov3.yolov4 import *
from tensorflow.python.saved_model import tag_constants
from numpy import mean
from PyQt5 import QtCore, QtGui, QtWidgets

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# connection = TCI.TwitchChatIRC()

# message = 'Hello everyone! I\'m here to make guesses!'
# connection.send('BirdBotML', message)

cwd = os.getcwd()

def load_yolo_weights(model, weights_file):
    tf.keras.backend.clear_session() # used to reset layer names
    # load Darknet original weights to TensorFlow model
    if YOLO_TYPE == "yolov3":
        range1 = 75 if not TRAIN_YOLO_TINY else 13
        range2 = [58, 66, 74] if not TRAIN_YOLO_TINY else [9, 12]
    if YOLO_TYPE == "yolov4":
        range1 = 110 if not TRAIN_YOLO_TINY else 21
        range2 = [93, 101, 109] if not TRAIN_YOLO_TINY else [17, 20]
    
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                # darknet weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                # tf weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'

def Load_Yolo_model():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
        
    if YOLO_FRAMEWORK == "tf": # TensorFlow detection
        if YOLO_TYPE == "yolov4":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        if YOLO_TYPE == "yolov3":
            Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
            
        if YOLO_CUSTOM_WEIGHTS == False:
            print("Loading Darknet_weights from:", Darknet_weights)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
            load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
        else:
            checkpoint = f"./checkpoints/{TRAIN_MODEL_NAME}"
            if TRAIN_YOLO_TINY:
                checkpoint += "_Tiny"
            print("Loading custom weights from:", checkpoint)
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(checkpoint)  # use custom weights
        
    elif YOLO_FRAMEWORK == "trt": # TensorRT detection
        saved_model_loaded = tf.saved_model.load(YOLO_CUSTOM_WEIGHTS, tags=[tag_constants.SERVING])
        signature_keys = list(saved_model_loaded.signatures.keys())
        yolo = saved_model_loaded.signatures['serving_default']

    return yolo

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
    
def draw_hats(image, bboxes, random_file_array, hatCounter, CLASSES=YOLO_COCO_CLASSES, rectangle_colors='', randomFirstTime=''):

    # print("Random pixel: " + str(random_hat[1][1]))
    
    for i, bbox in enumerate(bboxes):        
        # print("This is I: " + str(i))
        # print(hatCounter)
        if randomFirstTime == True or i > hatCounter:
           random_file_array.append(random.choice(os.listdir("B:\BirdBot\BirdKeras\TensorFlow2\hats")))
           hatCounter += 1
           print(random_file_array)
           print(randomFirstTime)
    
        hat_file = r'B:\BirdBot\BirdKeras\TensorFlow2\hats\\' + random_file_array[i]
        hat = cv2.imread(hat_file, -1)
        
        coor = np.array(bbox[:4], dtype=np.int32)
        
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        
        bhWidth = int(x2 - x1)
        # print(bhWidth)
        bhHigh = int(y2 - y1)
        # print(bhHigh)
        
        resized_hat = cv2.resize(hat, (bhWidth, bhHigh), interpolation = cv2.INTER_AREA)
        
        y = 0
        x = 0
        
        while y < bhHigh:
            # print("This is Y: " + str(y))
            while x < bhWidth:
                if resized_hat[y][x][0] > 200 and resized_hat[y][x][1] > 200 and resized_hat[y][x][2] > 200:
                    try:
                        resized_hat[y][x] = 0
                    except:
                        print("An exception occurred")
                x += 1
            y += 1
            x = 0
        
        y = 0
        x = 0
        
        #FIND A WAY TO LOOP THROUGH PIXELS TO DELETE WHITE SPACE.
        
        # print(resized_hat)
        
        img2gray = cv2.cvtColor(resized_hat, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        
        # print("Resized pixel: " + str(resized_hat[1][1]))
        
        try:
            roi = image[y1-45:y2-45, x1+5:x2+5]
        
            roi[np.where(mask)] = 0
            roi += resized_hat[:, :, :3]
        except:
            print("An exception occurred")
            
    randomFirstTime = False
        
    return image, hatCounter
    
def draw_bbox(image, bboxes, CLASSES=YOLO_COCO_CLASSES, show_label=True, draw_rect=False, show_confidence = True, Text_colors=(255,255,255), rectangle_colors='', tracking=False):   
    NUM_CLASS = read_class_names(CLASSES)
    num_classes = len(NUM_CLASS)
    image_h, image_w, _ = image.shape
    scientific_name = ''
    IUCN = ''
    global LABEL
    LABEL = []
    
    blue = []
    green = []
    red = []

    for i, bbox in enumerate(bboxes):        
        
        coor = np.array(bbox[:4], dtype=np.int32)
        # print(i)
        
        score = bbox[4]
        # print(score)
        
        class_ind = int(bbox[5])
        # print(class_ind)
        
        # print(str(NUM_CLASS[class_ind]))
        
        if NUM_CLASS[class_ind] == 'American Crow':
            rectangle_colors = (30,30,30)
            scientific_name = 'Corvus brachyrhynchos'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'American Goldfinch':
            rectangle_colors = (150,255,255)
            scientific_name = 'Spinus tristis'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'American Robin':
            rectangle_colors = (54,101,163)
            scientific_name = 'Turdus migratorius'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Bald Eagle':
            rectangle_colors = (25,95,130)
            scientific_name = 'Haliaeetus leucocephalus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Black Vulture':
            rectangle_colors = (25,30,30)
            scientific_name = 'Coragyps atratus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Black-capped Chickadee':
            rectangle_colors = (23,29,29)
            scientific_name = 'Poecile atricapillus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Black-crested Titmouse':
            rectangle_colors = (90,90,90)
            scientific_name = 'Baeolophus atricristatus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Black-headed Grosbeak':
            rectangle_colors = (0,106,188)
            scientific_name = 'Pheucticus melanocephalus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Black-headed Grosbeak - Female':
            rectangle_colors = (50,150,220)
            scientific_name = 'Pheucticus melanocephalus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Blue Jay':
            rectangle_colors = (160,95,35)
            scientific_name = 'Cyanocitta cristata'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Carolina Wren':
            rectangle_colors = (75,135,180)
            scientific_name = 'Thryothorus ludovicianus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Chestnut-backed Chickadee':
            rectangle_colors = (24,70,87)
            scientific_name = 'Poecile rufescens'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Chipping Sparrow':
            rectangle_colors = (50,110,155)
            scientific_name = 'Spizella passerina'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Dark-eyed Junco':
            rectangle_colors = (57,47,45)
            scientific_name = 'Junco hyemalis'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Downy Woodpecker':
            rectangle_colors = (56,69,79)
            scientific_name = 'Dryobates pubescens'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Eastern Bluebird':
            rectangle_colors = (235,155,73)
            scientific_name = 'Sialia sialis'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Eastern Phoebe':
            rectangle_colors = (125,125,125)
            scientific_name = 'Sayornis phoebe'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Eastern Towhee':
            rectangle_colors = (15,105,200)
            scientific_name = 'Pipilo erythrophthalmus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'European Starling':
            rectangle_colors = (96,96,96)
            scientific_name = 'Sturnus vulgaris'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Golden-crowned Sparrow':
            rectangle_colors = (0,215,215)
            scientific_name = 'Zonotrichia atricapilla'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Grey Catbird':
            rectangle_colors = (175,175,175)
            scientific_name = 'Dumetella carolinensis'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'House Finch - Male':
            rectangle_colors = (212,170,255)
            scientific_name = 'Haemorhous mexicanus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'House Finch - Female':
            rectangle_colors = (126,139,155)
            scientific_name = 'Haemorhous mexicanus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'House Sparrow':
            rectangle_colors = (10,66,89)
            scientific_name = 'Passer domesticus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Indigo Bunting':
            rectangle_colors = (0,95,190)
            scientific_name = 'Passerina cyanea'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Mourning Dove':
            rectangle_colors = (205,175,85)
            scientific_name = 'Zenaida macroura'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Northern Cardinal':
            rectangle_colors = (30,30,235)
            scientific_name = 'Cardinalis cardinalis'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Northern Flicker':
            rectangle_colors = (138,144,177)
            scientific_name = 'Colaptes auratus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Northern Mockingbird':
            rectangle_colors = (204,204,204)
            scientific_name = 'Mimus polyglottos'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Osprey':
            rectangle_colors = (66,98,117)
            scientific_name = 'Pandion haliaetus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Osprey - Chick':
            rectangle_colors = (121,142,155)
            scientific_name = 'Pandion haliaetus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Osprey - Egg':
            rectangle_colors = (145,176,196)
            scientific_name = 'Pandion haliaetus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Pine Siskin':
            rectangle_colors = (137,110,97)
            scientific_name = 'Spinus pinus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Pine Warbler':
            rectangle_colors = (0,255,255)
            scientific_name = 'Setophaga pinus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Purple Finch':
            rectangle_colors = (117,30,220)
            scientific_name = 'Haemorhous purpureus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Red-breasted Nuthatch':
            rectangle_colors = (58,95,196)
            scientific_name = 'Sitta canadensis'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Red-winged Blackbird':
            rectangle_colors = (25,25,25)
            scientific_name = 'Agelaius phoeniceus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Ruby-crowned Kinglet':
            rectangle_colors = (0,128,128)
            scientific_name = 'Regulus calendula'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Ruby-throated Hummingbird':
            rectangle_colors = (8,214,8)
            scientific_name = 'Archilochus colubris'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Steller\'s Jay':
            rectangle_colors = (191,95,0)
            scientific_name = 'Cyanocitta stelleri'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Spotted Towhee':
            rectangle_colors = (47,38,29)
            scientific_name = 'Pipilo maculatus'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Song Sparrow':
            rectangle_colors = (10,255,0)
            scientific_name = 'Melospiza melodia'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Townsend\'s Warbler':
            rectangle_colors = (53,214,255)
            scientific_name = 'Setophaga townsendi'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Tufted Titmouse':
            rectangle_colors = (75,75,75)
            scientific_name = 'Baeolophus bicolor'
            IUCN = "Least Concern"
        elif NUM_CLASS[class_ind] == 'Turkey Vulture':
            rectangle_colors = (0,95,142)
            scientific_name = 'Cathartes aura'
            IUCN = "Least Concern"
        else:
            rectangle_colors = (255,255,255)
        
        if sum(rectangle_colors) >= 300:
            blue.append(0)
            green.append(0)
            red.append(0)
        else: 
            blue.append(255)
            green.append(255)
            red.append(255)
        
        bbox_color = rectangle_colors
        bbox_thick = int(0.5 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        if draw_rect:
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " - {:.1f}%".format(score*100) if show_confidence else ""

            if tracking: score_str = " "+str(score)

            try:
                label = "{}".format(NUM_CLASS[class_ind]) + score_str
                LABEL.append(label)
                label_sci = scientific_name + " - " + IUCN
            except KeyError:
                print("You received KeyError, this might be that you are trying to use yolo original weights")
                print("while using custom classes, if using custom model in http://configs.py set YOLO_CUSTOM_WEIGHTS = True")

            # get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            (text_width_sci, text_height_sci), baseline_sci = cv2.getTextSize(label_sci, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                  fontScale, thickness=bbox_thick)
            # put filled text rectangle
            cv2.rectangle(image, (x1, y1-text_height-24), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
            cv2.rectangle(image, (x1, y1), (x1 + text_width_sci, y1 - text_height_sci - baseline_sci), bbox_color, thickness=cv2.FILLED)
           
            Text_colors=(blue[i], green[i], red[i])
            
            # put text above rectangle
            cv2.putText(image, label, (x1, y1-text_height-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

            cv2.putText(image, label_sci, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0 
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold 
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


def detect_image(Yolo, image_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.5, iou_threshold=0.3, rectangle_colors=''):
    
    write_label = ''

    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    if YOLO_FRAMEWORK == "tf":
        pred_bbox = Yolo.predict(image_data)
    elif YOLO_FRAMEWORK == "trt":
        batched_input = tf.constant(image_data)
        result = Yolo(batched_input)
        pred_bbox = []
        for key, value in result.items():
            value = value.numpy()
            pred_bbox.append(value)
        
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')

    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, draw_rect=True, rectangle_colors=rectangle_colors)

    cv2.imwrite("temp.jpg", image)

    counter = 1

    for x in LABEL:
        if len(LABEL) > counter:
            write_label += "ID #" + str(counter) + " - " + x + "\n"
            print("counter " + str(counter))
            print("length " + str(len(LABEL)))
            counter += 1
        else: 
            write_label += "ID #" + str(counter) + " - " + x


    with open('label.txt', 'w') as label_file:
        label_file.write(write_label)
    print(write_label)
    # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        
    return LABEL

def Predict_bbox_mp(Frames_data, Predicted_data, Processing_times):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")
    Yolo = Load_Yolo_model()
    times = []
    while True:
        if Frames_data.qsize()>0:
            image_data = Frames_data.get()
            t1 = time.time()
            Processing_times.put(time.time())
            
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            
            Predicted_data.put(pred_bbox)


def postprocess_mp(Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES, score_threshold, iou_threshold, rectangle_colors, realtime):
    times = []
    while True:
        if Predicted_data.qsize()>0:
            pred_bbox = Predicted_data.get()
            if realtime:
                while original_frames.qsize() > 1:
                    original_image = original_frames.get()
            else:
                original_image = original_frames.get()
            
            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')
            image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            times.append(time.time()-Processing_times.get())
            times = times[-20:]
            
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
            #print("Time: {:.2f}ms, Final FPS: {:.1f}".format(ms, fps))
            
            Processed_frames.put(image)

def Show_Image_mp(Processed_frames, show, Final_frames):
    while True:
        if Processed_frames.qsize()>0:
            image = Processed_frames.get()
            Final_frames.put(image)
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

def Send_message(message):
    s.send("PRIVMSG #YOURCHANNELNAME :" + message + "\r\n")

# detect from webcam
def detect_video_realtime_mp(video_path, output_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', realtime=False):
    if realtime:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    no_of_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    original_frames = Queue()
    Frames_data = Queue()
    Predicted_data = Queue()
    Processed_frames = Queue()
    Processing_times = Queue()
    Final_frames = Queue()
    
    p1 = Process(target=Predict_bbox_mp, args=(Frames_data, Predicted_data, Processing_times))
    p2 = Process(target=postprocess_mp, args=(Predicted_data, original_frames, Processed_frames, Processing_times, input_size, CLASSES, score_threshold, iou_threshold, rectangle_colors, realtime))
    p3 = Process(target=Show_Image_mp, args=(Processed_frames, show, Final_frames))
    p1.start()
    p2.start()
    p3.start()
        
    while True:
        ret, img = vid.read()
        if not ret:
            break

        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_frames.put(original_image)

        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        Frames_data.put(image_data)
        
    while True:
        if original_frames.qsize() == 0 and Frames_data.qsize() == 0 and Predicted_data.qsize() == 0  and Processed_frames.qsize() == 0  and Processing_times.qsize() == 0 and Final_frames.qsize() == 0:
            p1.terminate()
            p2.terminate()
            p3.terminate()
            break
        elif Final_frames.qsize()>0:
            image = Final_frames.get()
            if output_path != '': out.write(image)

    cv2.destroyAllWindows()

def detect_video(Yolo, video_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, score_threshold=0.5, iou_threshold=0.3, rectangle_colors='', HatMode=''):
    
    global randomFirstTime
    global random_file_array
    global hatCounter
    
    random_file_array = []
    randomFirstTime = True
    hatCounter = -1
    
    print(video_path)
    print("Hat Mode: " + str(HatMode))
    
    times, times_2 = [], []
    vid = cv2.VideoCapture(video_path)
    
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    if WRITE_VIDEO_OUTPUT_FILE:
        out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    
    if WRITE_VIDEO_TRAIN_FILE:
        train = cv2.VideoWriter(str(output_path[:-4] + '-Train.mp4'), codec, fps, (width, height)) # output_path must be .mp4
        
    if WRITE_VIDEO_GUESS_FILE:
        guess = cv2.VideoWriter(str(output_path[:-4] + '-Guess.mp4'), codec, fps, (width, height)) # output_path must be .mp4
    
    frameCount = 0
    videoRecordIncrementor = 0
    birdCountMax = 0
    birdsEarned = 0
    min_frames_thresh = 60
    frameThreshold = 90
    species_frame_count = 0
    seen_species_array = []
    wrapped_seen_species = ''
    approval_species_array = []
    temp_species_array = []
    currentScore = []
    timer_species_array = []
    
    t = time.localtime()
    
    bbox_thick = int(0.6 * (height + width) / 1000)
    
    if bbox_thick < 1:
        bbox_thick = 1
    
    fontScale = 0.75 * bbox_thick
    
    NUM_CLASS = read_class_names(CLASSES)
    
    _, img2 = vid.read()
    
    blk = np.zeros(img2.shape, np.uint8)
    
    cv2.rectangle(blk, (20, height-20), (125, height-50), (255, 255, 255), cv2.FILLED)
    
    cv2.rectangle(blk, (width-225, height-20), (width-20, height-50), (255, 255, 255), cv2.FILLED)

    while True:
        _, img = vid.read()
        
        guess_copy = img
        currentSpecies = ''
        
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
        csv_date = time.strftime("%m-%d-%Y", t)
        csv_time = time.strftime("%H:%M:%S", t)
        
        if img is None:
            if WRITE_VIDEO_OUTPUT_FILE:
                out.release()
            if WRITE_VIDEO_TRAIN_FILE:
                train.release()
            if WRITE_VIDEO_GUESS_FILE:
                guess.release()
                os.rename(output_path[:-4] + '-Guess.mp4', 'YOLO_Videos/' + max(seen_species_array) + '-Guess' + str(videoRecordIncrementor) + '.mp4')
            cv2.destroyAllWindows()
            print("END OF VIDEO")
            break
        
        # blk = np.zeros(img.shape, np.uint8)
        
        # cv2.rectangle(blk, (20, height-20), (160, height-50), (255, 255, 255), cv2.FILLED)
        
        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')        
        
        if HatMode == True:
            image, hatCounter = draw_hats(original_image, bboxes, random_file_array, hatCounter, CLASSES=CLASSES, rectangle_colors=rectangle_colors, randomFirstTime=randomFirstTime)
        else:
            image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        
        for i, bbox in enumerate(bboxes):
            
            currentSpecies = NUM_CLASS[int(bbox[5])]
            currentScore.append(float(bbox[4]))
            randomFirstTime = False

            approval_species_array.append(currentSpecies)
            
            if len(temp_species_array) <= i or (approval_species_array.count(currentSpecies) >= min_frames_thresh and currentSpecies not in temp_species_array):
                temp_species_array.append(currentSpecies)
                
                birdSpeciesCount = len(temp_species_array)
                birdsTokenCalc = birdSpeciesCount * 5
                birdsEarned += birdsTokenCalc
            
            if currentSpecies in approval_species_array:
                
                if videoRecordIncrementor < frameCount:
                    start_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
                
                species_frame_count = approval_species_array.count(currentSpecies)
                lastMaxSpecies = max(approval_species_array)
                videoRecordIncrementor = frameCount + 60
                
                if species_frame_count >= min_frames_thresh and currentSpecies not in seen_species_array:

                    seen_species_array.append(currentSpecies)
                    wrapped_seen_species = textwrap.wrap('Today\'s Seen Species: '+str(seen_species_array), width=width/20)
                    
                    print (currentSpecies + ' added to today\'s seen species')
        
        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

            gap = textsize[1] + 16

            y = 45 + i * gap
            x = 40 + textsize[0]
            
            cv2.rectangle(blk, (20, 20), (x, y+10), (255, 255, 255), cv2.FILLED)

            cv2.rectangle(blk, (20, y), (215, y+38), (255, 255, 255), cv2.FILLED)            
        
        
        # Create opacity overlay        
        image = cv2.addWeighted(image, 1.0, blk, 0.40, 1)

        birdCount = len(bboxes)
        
        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

            gap = textsize[1] + 16

            y = 45 + i * gap
            x = int((image.shape[1] - textsize[0]) / 2)
        
            cv2.putText(image, line, (30, y-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
            
            cv2.putText(image, "Bird Count [" + str(birdCountMax) + "]: " + str(birdCount), (30, y+28), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
        
        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-300:]
        times_2 = times_2[-300:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        frameCount += 1
        
        # Put FPS on screen
        image = cv2.putText(image, "FPS: {:.1f}".format(fps2), (30, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        
        # Put Time on screen
        image = cv2.putText(image, current_time, (width-215, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        
        # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        
        # The conditional to write a train file which is anytime the object detector doesn't see anything
        if not currentSpecies and WRITE_VIDEO_TRAIN_FILE:
            train.write(image)
        
        # The conditionals to write guess videos which is when the object detector sees something plus 2 seconds
        if videoRecordIncrementor > frameCount and WRITE_VIDEO_GUESS_FILE:
            guess.write(guess_copy)
        elif videoRecordIncrementor == frameCount and WRITE_VIDEO_GUESS_FILE:
            guess.release()
            timeWithoutDate = str(current_time[-11:])
            timeWithoutDate = timeWithoutDate.replace(':','-')
            os.rename(output_path[:-4] + '-Guess.mp4', 'YOLO_Videos/' + lastMaxSpecies + '-Guess' + '-' + str(current_time[:-9]) + '-' + timeWithoutDate + '.mp4')
            guess = cv2.VideoWriter(str(output_path[:-4] + '-Guess.mp4'), codec, fps, (width, height))

            # Reset temp variables
            currentScore = []
            temp_species_array =[]
            
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
                
        if frameCount % frameThreshold == 0:
            approval_species_array = []
            species_frame_count = 0

    cv2.destroyAllWindows()

# detect from webcam
def detect_realtime(Yolo, output_path, camera_id, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.5, iou_threshold=0.3, rectangle_colors='', ALGORAND_WALLET=''):
    times, times_2 = [], []
	
    vid = cv2.VideoCapture(int(camera_id))

    # by default VideoCapture returns float instead of int
    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    width = 1920
    print(width)
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = 1080
    print(height)
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    frameCount = 0
    totalDailySightingFormula = "=COUNTIF($A$2:$A$500000,INDIRECT(ADDRESS(ROW(),1)))"
    totalSightingsFormula = "=COUNTA($D$2:$D$500000)"
    totalUnsuccessfulFormula = "=COUNTIF($D$2:$D$500000,\"[]\")"
    rateOfSuccess = "=(INDIRECT(ADDRESS(ROW(),8))-INDIRECT(ADDRESS(ROW(),9)))/INDIRECT(ADDRESS(ROW(),8))"
    videoRecordIncrementor = 0
    g = geocoder.ip('me')
    geolocator = Nominatim(user_agent="http")
    cords = str(g.lat) + "," + str(g.lng)
    loc = geolocator.reverse(cords)
    city = loc.raw.get('address').get('city')
    min_frames_thresh = 10
    frameThreshold = 15
    species_frame_count = 0
    birdsEarned = 0
    seen_species_array = []
    wrapped_seen_species = ''
    approval_species_array = []
    temp_species_array = []
    last_temp_species_array = "Last seen: []"
    last_array_x = 160
    last_time_seen = ""
    currentScore = []
    timer_species_array = []
    first_guess = True
    
    t = time.localtime()
    
    start_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
    
    bbox_thick = int(0.5 * (height + width) / 1000)
    
    if bbox_thick < 1:
        bbox_thick = 1
    
    fontScale = 0.75 * bbox_thick
    
    NUM_CLASS = read_class_names(CLASSES)
    
    _, img2 = vid.read()
    
    img2 = cv2.resize(img2, (width, height))
    
    blk = np.zeros(img2.shape, np.uint8)
    
    cv2.rectangle(blk, (20, height-20), (125, height-50), (255, 255, 255), cv2.FILLED)
    
    cv2.rectangle(blk, (width-225, 50), (width-20, 80), (255, 255, 255), cv2.FILLED)
    
    cv2.rectangle(blk, (width-225, 20), (width-20, 50), (255, 255, 255), cv2.FILLED)

    while True:
        _, image = vid.read()
        
        image = cv2.resize(image, (width, height))
        
        guess_copy = image
        currentSpecies = ''
        
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
        csv_date = time.strftime("%m-%d-%Y", t)
        csv_time = time.strftime("%H:%M:%S", t)
        file_time = time.strftime("%H-%M-%S", t)
        current_time_delta = time.time()
        
        if image is None:
            cv2.destroyAllWindows()
            print("END OF VIDEO")
            break
        
        # blk = np.zeros(image.shape, np.uint8)
        
        # cv2.rectangle(blk, (20, height-20), (160, height-50), (255, 255, 255), cv2.FILLED)
        
        try:
            original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')        
        
        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        
        birdCount = len(bboxes)
        
        for i, bbox in enumerate(bboxes):
            
            currentSpecies = NUM_CLASS[int(bbox[5])]
            currentScore.append(float(bbox[4]))

            approval_species_array.append(currentSpecies)
            
            if (len(temp_species_array) <= i and approval_species_array.count(currentSpecies) >= min_frames_thresh) or (currentSpecies not in temp_species_array and approval_species_array.count(currentSpecies) >= min_frames_thresh):
                
                tempPhotoName = ALGORAND_WALLET[0:10] + "-" + str(currentSpecies.replace(" ", "-")) + "-" + str(file_time) + ".jpg"
                tempPhotoLoc = cwd + "/" + ALGORAND_WALLET[0:10] + "-" + str(currentSpecies.replace(" ", "-")) + "-" + str(file_time) + ".jpg"
                        
                cv2.imwrite(tempPhotoLoc, guess_copy)
                
                with open(tempPhotoName, "rb") as img_file:
                    BirdBotPhotoBase64 = base64.b64encode(img_file.read()).decode('utf-8')
                    img_file.close()
                    os.remove(tempPhotoLoc)
                # print(BirdBotPhotoBase64)
                
                temp_species_array.append(currentSpecies)
                last_temp_species_array = "Last seen: " + str(temp_species_array)
                last_array_textsize = cv2.getTextSize(last_temp_species_array, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]
                last_array_x = 40 + last_array_textsize[0]
                
                blk = np.zeros(img2.shape, np.uint8)
                
                last_time_seen = current_time
                cv2.rectangle(blk, (width-225, height-50), (width-20, height-80), (255, 255, 255), cv2.FILLED)               
                cv2.rectangle(blk, (20, height-20), (125, height-50), (255, 255, 255), cv2.FILLED)
                cv2.rectangle(blk, (width-225, 20), (width-20, 50), (255, 255, 255), cv2.FILLED)
                cv2.rectangle(blk, (width-225, 50), (width-20, 80), (255, 255, 255), cv2.FILLED)
                
                birdCountMax = birdCount
            
                url = POWER_URL
            
                BirdBotHeaders = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                
                start_time_delta = time.time()
                
                durationSighting = current_time_delta - start_time_delta
                
                BirdBotData = {
                "Date": str(csv_date),
                "Time": str(csv_time),
                "BIRDS_Earned": str(birdsEarned),
                "Algorand_Wallet": str(ALGORAND_WALLET),
                "BirdBot_Camera": str(BIRDBOT_CAMERA_NAME),
                "Bird_Species_Triggered": str(currentSpecies),
                "Bird_Photo": str(BirdBotPhotoBase64),
                "Bird_Species_Array": str(temp_species_array),
                "Approximate_Lat": str(g.lat),
                "Approximate_Long": str(g.lng),
                "Approximate_Location": str(loc),
                "Approximate_City": str(city),
                "Number_of_Birds": str(birdCount),
                "Duration_of_Sighting_(Seconds)": str(durationSighting)
                }

                BirdBotDataLog = requests.post(url, json=BirdBotData, headers=BirdBotHeaders)
                
            if currentSpecies in approval_species_array:
                
                if videoRecordIncrementor < frameCount:
                    start_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
                    start_time_delta = time.time()
                    
                species_frame_count = approval_species_array.count(currentSpecies)
                lastMaxSpecies = max(approval_species_array)
                videoRecordIncrementor = frameCount + 90
                
                if species_frame_count >= min_frames_thresh and currentSpecies not in seen_species_array:

                    seen_species_array.append(currentSpecies)
                    wrapped_seen_species = textwrap.wrap('Today\'s Seen Species: '+str(seen_species_array), width=width/20)
                    
                    print(currentSpecies + ' added to today\'s seen species')
        
        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

            gap = textsize[1] + 16

            y = 45 + i * gap
            x = 40 + textsize[0]
            
            cv2.rectangle(blk, (20, 20), (x, y+10), (255, 255, 255), cv2.FILLED)    
        
        cv2.rectangle(blk, ((width-(last_array_x+20)), height-20), (width-20, height-50), (255, 255, 255), cv2.FILLED)
        
        # Create opacity overlay        
        image = cv2.addWeighted(image, 1.0, blk, 0.40, 1)
        
        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

            gap = textsize[1] + 16

            y = 45 + i * gap
            x = int((image.shape[1] - textsize[0]) / 2)
        
            cv2.putText(image, line, (30, y-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
        
        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-60:]
        times_2 = times_2[-60:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        frameCount += 1
        
        # Put FPS on screen
        image = cv2.putText(image, "FPS: {:.1f}".format(fps2), (30, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        
        # Put BIRDS Earned on screen
        image = cv2.putText(image, "BIRDS Earned: " + str(birdsEarned), (width-215, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        
        # Put time on screen
        image = cv2.putText(image, current_time, (width-215, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        
        # Put last Species Array on screen
        image = cv2.putText(image, str(last_temp_species_array), (width-last_array_x, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        
        # Put last time seen on screen
        image = cv2.putText(image, last_time_seen, (width-215, height-60), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
 
        
        # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        
        # The conditionals to write guess videos which is when the object detector sees something plus 2 seconds
            
        if videoRecordIncrementor == frameCount and WRITE_VIDEO_GUESS_FILE:
            
            birdSpeciesCount = len(temp_species_array)
            birdsTokenCalc = birdSpeciesCount * 5
            birdsEarned += birdsTokenCalc
            
            if birdsEarned >= 500:
                birdsEarned = 500
            
            timeWithoutDate = str(current_time[-11:])
            timeWithoutDate = timeWithoutDate.replace(':','-')
            
            # connection = TCI.TwitchChatIRC()
            # TWITCH_SEEN_MESSAGE = 'BirdBot is ' + str("{:.2f}".format(mean(currentScore) * 100)) + '% confident it saw ' + str(temp_species_array) + ' from ' + str(start_time) + ' to ' + str(current_time)
            # connection.send('BirdBotML', TWITCH_SEEN_MESSAGE)
            # connection.close_connection()
                
            # Reset temp variables
            currentScore = []
            temp_species_array =[]
            first_guess = False

        if show:
            window_name = 'BirdBot - Real-Time'
            cv2.imshow(window_name, image)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
                
        if frameCount % frameThreshold == 0:
            approval_species_array = []
            species_frame_count = 0

        if csv_time == "05:30:00":
            birdsEarned = 0
            seen_species_array = []

    cv2.destroyAllWindows()
    
# detect from IP camera
def detect_ip_camera(Yolo, output_path, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.6, iou_threshold=0.35, rectangle_colors='', ALGORAND_WALLET='', IP_CAMERA_NAME=''):
    
    times, times_2 = [], []
    now = datetime.now()
    current_time = now.strftime("%m-%d-%M")
    
    video_stream_widget = RTSPVideoWriterObject(IP_CAMERA_NAME)
    
    frameCount = 0
    totalDailySightingFormula = "=COUNTIF($A$2:$A$500000,INDIRECT(ADDRESS(ROW(),1)))"
    totalSightingsFormula = "=COUNTA($D$2:$D$500000)"
    totalUnsuccessfulFormula = "=COUNTIF($D$2:$D$500000,\"[]\")"
    rateOfSuccess = "=(INDIRECT(ADDRESS(ROW(),8))-INDIRECT(ADDRESS(ROW(),9)))/INDIRECT(ADDRESS(ROW(),8))"
    videoRecordIncrementor = 0
    g = geocoder.ip('me')
    geolocator = Nominatim(user_agent="http")
    cords = str(g.lat) + "," + str(g.lng)
    loc = geolocator.reverse(cords)
    city = loc.raw.get('address').get('city')
    min_frames_thresh = 10
    frameThreshold = 15
    species_frame_count = 0
    birdsEarned = 0
    seen_species_array = []
    wrapped_seen_species = ''
    approval_species_array = []
    temp_species_array = []
    last_temp_species_array = "Last seen: []"
    last_array_x = 160
    last_time_seen = ""
    currentScore = []
    timer_species_array = []
    first_guess = True
    
    t = time.localtime()
    
    width = 1080
    height = 1920
    fps = 30
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    
    bbox_thick = int(0.6 * (height + width) / 1000)
    
    if bbox_thick < 1:
        bbox_thick = 1
    
    fontScale = 0.75 * bbox_thick
    
    NUM_CLASS = read_class_names(CLASSES)

    while True:
            
        times, times_2 = [], []
        t = time.localtime()
        t1 = time.time()
    
        time.sleep(.00001)   # simulate time between events
    
        try:
            video_stream_widget.show_frame(Yolo, score_threshold, iou_threshold, CLASSES, rectangle_colors)
            # video_stream_widget.save_frame()
        except AttributeError:
            pass
    
        t2 = time.time()
        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)

        times = times[-60:]
        times_2 = times_2[-60:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))        
        
        # print(output_path)
                
        if frameCount % frameThreshold == 0:
            approval_species_array = []
            species_frame_count = 0
    
    cap_audio.terminate()
    cv2.destroyAllWindows()
    
# bufferless VideoCapture
class RTSPVideoWriterObject(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(IP_CAMERA_NAME)
        self.cap_audio = subprocess.Popen('vlc --novideo -Idummy '+str(IP_CAMERA_NAME))
        time.sleep(2.6)

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        # self.output_video = cv2.VideoWriter('output.avi', self.codec, 30, (self.frame_width, self.frame_height))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self, Yolo, score_threshold, iou_threshold, CLASSES, rectangle_colors):
        # Display frames in main program
        if self.status:
            
            image_data = image_preprocess(np.copy(self.frame), [YOLO_INPUT_SIZE, YOLO_INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)
            
            t2 = time.time()
            
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, self.frame, YOLO_INPUT_SIZE, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')        
            
            image = draw_bbox(self.frame, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            cv2.imshow('frame', image)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.cap_audio.terminate()
            # self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

# Generate data from a machine learning model
def generate_ml_data(Yolo, video_path, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, score_threshold=0.6, iou_threshold=0.5, rectangle_colors=''):

    print(video_path)
    
    source = "B:/BirdBot/BirdModelTraining/train_images"

    videoFiles = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.mp4)$', f)]

    print(videoFiles)

    now = datetime.now()
    current_time = now.strftime("%m-%d-%M")

    currentframe = 0
    countframes = 60
    total_count = 0
    currentSpecies = ''

    for file in videoFiles:
        
        videoName = file
        print(videoName)
        vid = cv2.VideoCapture('B:/BirdBot/BirdModelTraining/train_images/'+videoName)
        
        writeFileName = videoName[:-4]
        
        # by default VideoCapture returns float instead of int
        imW = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        imH = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        NUM_CLASS = read_class_names(CLASSES)
        
        while vid.isOpened():
            
            ret, frame = vid.read()
            trainframe = frame
            object_count = 0
            currentframe += 1
            
            if currentframe % countframes == 0:
                if ret:
                    if frame is None:
                        cv2.destroyAllWindows()
                        # print("END OF VIDEO")
                        break
                    
                    try:
                        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    except:
                        break
                    
                    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
                    image_data = image_data[np.newaxis, ...].astype(np.float32)

                    if YOLO_FRAMEWORK == "tf":
                        pred_bbox = Yolo.predict(image_data)
                    elif YOLO_FRAMEWORK == "trt":
                        batched_input = tf.constant(image_data)
                        result = Yolo(batched_input)
                        pred_bbox = []
                        for key, value in result.items():
                            value = value.numpy()
                            pred_bbox.append(value)
                    
                    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                    pred_bbox = tf.concat(pred_bbox, axis=0)

                    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
                    bboxes = nms(bboxes, iou_threshold, method='nms')        
                    
                    objectHolder = {}
                    
                    root = ET.Element("annotation")
                    
                    # Write training photo every 5 frames on successful ID
                        
                    folder = ET.SubElement(root, "folder")
                    folder.text = "images"
                    filename = ET.SubElement(root, "filename") 
                    filename.text = writeFileName + "-" + str(currentframe) + ".jpg"
                    path = ET.SubElement(root, "path")  
                    path.text = "images/"+writeFileName + "-" + str(currentframe) + ".jpg"
                    source = ET.SubElement(root, "source")
                    
                    database = ET.SubElement(source, "database")
                    database.text = "Unspecified"
                    
                    size = ET.SubElement(root, "size")
                    
                    width = ET.SubElement(size, "width")
                    width.text = str(imW)
                    height = ET.SubElement(size, "height")
                    height.text = str(imH)
                    depth = ET.SubElement(size, "depth")
                    depth.text = str(3)
                    
                    # print(str(len(bboxes)))
                    
                    for i, bbox in enumerate(bboxes):
                        
                        # print(bbox)
                        
                        coor = np.array(bbox[:4], dtype=np.int32)
                        
                        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
                        
                        # print(coor)
                        
                        currentSpecies = NUM_CLASS[int(bbox[5])]
                        
                        object_name = currentSpecies
                        
                        ymin = y1
                        # print(ymin)
                        xmin = x1
                        # print(xmin)
                        ymax = y2
                        # print(ymax)
                        xmax = x2
                        # print(xmax)
                        
                        print(currentSpecies + " - Count: " + str(total_count) + " - Frame: " + str(currentframe))
                        
                        objectHolder["objectXml"+str(i)] = ET.SubElement(root, "object")
                            
                        objectHolder["nameXml"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "name")
                        objectHolder["nameXml"+str(i)].text = object_name
                        objectHolder["pose"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "pose")
                        objectHolder["pose"+str(i)].text = "Unspecified"
                        objectHolder["truncated"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "truncated") 
                        objectHolder["truncated"+str(i)].text = "Unspecified"
                        objectHolder["difficult"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "difficult")
                        objectHolder["difficult"+str(i)].text = "Unspecified"
                        
                        objectHolder["bndbox"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "bndbox") 
                        
                        objectHolder["xminXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "xmin")
                        objectHolder["xminXml"+str(i)].text = str(xmin)
                        objectHolder["yminXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "ymin") 
                        objectHolder["yminXml"+str(i)].text = str(ymin)
                        objectHolder["xmaxXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "xmax") 
                        objectHolder["xmaxXml"+str(i)].text = str(xmax)
                        objectHolder["ymaxXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "ymax") 
                        objectHolder["ymaxXml"+str(i)].text = str(ymax)
                        # print(objectHolder["ymaxXml"+str(i)].text)
                        
                        # print (ET.tostring(root))
                        
                        object_count += 1
                        total_count += 1
                        
                    if object_count > 0:
                        
                        name = "B:/BirdBot/BirdModelTraining/train_images/" + writeFileName + "-" + str(currentframe) + ".jpg"
                        
                        cv2.imwrite(name, trainframe)
                        
                        # print("Wrote: " + name)
                        
                        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
                        
                        xmlName = 'B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml'
                        
                        with open('B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml', "w") as f:
                            
                            f.write(xmlstr)
                            
                            # print("Wrote: " + xmlName)
                            
                        a_file = open('B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml', "r")
                            
                        lines = a_file.readlines()
                        a_file.close()
                        
                        del lines[0]
                        
                        new_file = open('B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml', "w+")
                        
                        for line in lines:
                            new_file.write(line)
                        
                        new_file.close()
                    
                else:
                    vid.release()
                    currentframe = 0
                    cv2.destroyAllWindows()
                    break
            
            elif frame is None:
                cv2.destroyAllWindows()
                # print("END OF VIDEO")
                break


    ##################################################
        
    ############ Start of the GUI Section ############ 
        
    ##################################################

        
def ProcessVideo_GUI(self, Yolo, video_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, score_threshold=0.5, iou_threshold=0.3, rectangle_colors=''):
    
    print(video_path)
    
    source = "B:/BirdBot/BirdModelTraining/train_images"

    videoFiles = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.mp4)$', f)]

    print(videoFiles)
    
    global randomFirstTime
    global random_file_array
    global hatCounter
    
    random_file_array = []
    randomFirstTime = True
    hatCounter = -1
    HatMode = False
	
    print(video_path)
    print("Hat Mode: " + str(HatMode))
    
    for file in videoFiles:
    
        times, times_2 = [], []
        videoName = file
        print(videoName)
        vid = cv2.VideoCapture('B:/BirdBot/BirdModelTraining/train_images/'+videoName)
        
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        
        if WRITE_VIDEO_OUTPUT_FILE:
            out = cv2.VideoWriter('B:/BirdBot/BirdModelTraining/train_images/'+videoName[:-4]+'-ML.mp4', codec, fps, (width, height)) # output_path must be .mp4
        
        if WRITE_VIDEO_TRAIN_FILE:
            train = cv2.VideoWriter(str(output_path[:-4] + '-Train.mp4'), codec, fps, (width, height)) # output_path must be .mp4
            
        if WRITE_VIDEO_GUESS_FILE:
            guess = cv2.VideoWriter(str(output_path[:-4] + '-Guess.mp4'), codec, fps, (width, height)) # output_path must be .mp4
        
        frameCount = 0
        videoRecordIncrementor = 0
        birdCountMax = 0
        birdsEarned = 0
        min_frames_thresh = 60
        frameThreshold = 90
        species_frame_count = 0
        seen_species_array = []
        wrapped_seen_species = ''
        approval_species_array = []
        temp_species_array = []
        currentScore = []
        timer_species_array = []
        
        t = time.localtime()
        
        bbox_thick = int(0.6 * (height + width) / 1000)
        
        if bbox_thick < 1:
            bbox_thick = 1
        
        fontScale = 0.75 * bbox_thick
        
        NUM_CLASS = read_class_names(CLASSES)
        
        _, img2 = vid.read()
        
        blk = np.zeros(img2.shape, np.uint8)
        
        cv2.rectangle(blk, (20, height-20), (125, height-50), (255, 255, 255), cv2.FILLED)
        
        cv2.rectangle(blk, (width-225, height-20), (width-20, height-50), (255, 255, 255), cv2.FILLED)

        while True:
            _, img = vid.read()
            
            guess_copy = img
            currentSpecies = ''
            
            t = time.localtime()
            current_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
            csv_date = time.strftime("%m-%d-%Y", t)
            csv_time = time.strftime("%H:%M:%S", t)
            
            if img is None:
                if WRITE_VIDEO_OUTPUT_FILE:
                    out.release()
                if WRITE_VIDEO_TRAIN_FILE:
                    train.release()
                if WRITE_VIDEO_GUESS_FILE:
                    guess.release()
                    os.rename(output_path[:-4] + '-Guess.mp4', 'YOLO_Videos/' + max(seen_species_array) + '-Guess' + str(videoRecordIncrementor) + '.mp4')
                cv2.destroyAllWindows()
                print("END OF VIDEO")
                break
            
            # blk = np.zeros(img.shape, np.uint8)
            
            # cv2.rectangle(blk, (20, height-20), (160, height-50), (255, 255, 255), cv2.FILLED)
            
            try:
                original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            except:
                break
            
            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            t1 = time.time()
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)
            
            t2 = time.time()
            
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
            bboxes = nms(bboxes, iou_threshold, method='nms')        
            
            if HatMode == True:
                image, hatCounter = draw_hats(original_image, bboxes, random_file_array, hatCounter, CLASSES=CLASSES, rectangle_colors=rectangle_colors, randomFirstTime=randomFirstTime)
            else:
                image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
            
            for i, bbox in enumerate(bboxes):
                
                currentSpecies = NUM_CLASS[int(bbox[5])]
                currentScore.append(float(bbox[4]))
                randomFirstTime = False

                approval_species_array.append(currentSpecies)
                
                if len(temp_species_array) <= i or (approval_species_array.count(currentSpecies) >= min_frames_thresh and currentSpecies not in temp_species_array):
                    temp_species_array.append(currentSpecies)
                    
                    birdSpeciesCount = len(temp_species_array)
                    birdsTokenCalc = birdSpeciesCount * 5
                    birdsEarned += birdsTokenCalc
                
                if currentSpecies in approval_species_array:
                    
                    if videoRecordIncrementor < frameCount:
                        start_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
                    
                    species_frame_count = approval_species_array.count(currentSpecies)
                    lastMaxSpecies = max(approval_species_array)
                    videoRecordIncrementor = frameCount + 60
                    
                    if species_frame_count >= min_frames_thresh and currentSpecies not in seen_species_array:

                        seen_species_array.append(currentSpecies)
                        wrapped_seen_species = textwrap.wrap('Today\'s Seen Species: '+str(seen_species_array), width=width/20)
                        
                        print (currentSpecies + ' added to today\'s seen species')
            
            for i, line in enumerate(wrapped_seen_species):
                            
                textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

                gap = textsize[1] + 16

                y = 45 + i * gap
                x = 40 + textsize[0]
                
                cv2.rectangle(blk, (20, 20), (x, y+10), (255, 255, 255), cv2.FILLED)

                cv2.rectangle(blk, (20, y), (215, y+38), (255, 255, 255), cv2.FILLED)            
            
            
            # Create opacity overlay        
            image = cv2.addWeighted(image, 1.0, blk, 0.40, 1)
            
            for i, line in enumerate(wrapped_seen_species):
                            
                textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

                gap = textsize[1] + 16

                y = 45 + i * gap
                x = int((image.shape[1] - textsize[0]) / 2)
            
                cv2.putText(image, line, (30, y-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
                
                cv2.putText(image, "Bird Count [" + str(birdCountMax) + "]: " + str(birdCount), (30, y+28), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
            
            birdCount = len(bboxes)
            
            if birdCount > birdCountMax:
            
                birdCountMax = birdCount
            
            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)
            
            times = times[-60:]
            times_2 = times_2[-60:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
            
            frameCount += 1
            
            # Put FPS on screen
            image = cv2.putText(image, "FPS: {:.1f}".format(fps2), (30, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
            
            # Put Time on screen
            image = cv2.putText(image, current_time, (width-215, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
            # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
            
            # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            
            # The conditional to write a train file which is anytime the object detector doesn't see anything
            if not currentSpecies and WRITE_VIDEO_TRAIN_FILE:
                train.write(image)
            
            # The conditionals to write guess videos which is when the object detector sees something plus 2 seconds
            if videoRecordIncrementor > frameCount and WRITE_VIDEO_GUESS_FILE:
                guess.write(guess_copy)
            elif videoRecordIncrementor == frameCount and WRITE_VIDEO_GUESS_FILE:
                guess.release()
                timeWithoutDate = str(current_time[-11:])
                timeWithoutDate = timeWithoutDate.replace(':','-')
                os.rename(output_path[:-4] + '-Guess.mp4', 'YOLO_Videos/' + lastMaxSpecies + '-Guess' + '-' + str(current_time[:-9]) + '-' + timeWithoutDate + '.mp4')
                guess = cv2.VideoWriter(str(output_path[:-4] + '-Guess.mp4'), codec, fps, (width, height))
                # TWITCH_SEEN_MESSAGE = 'BirdBot is ' + str("{:.2f}".format(mean(currentScore) * 100)) + '% confident it saw ' + str(temp_species_array) + ' from ' + str(start_time) + ' to ' + str(current_time)
                # connection.send('BirdBotML', TWITCH_SEEN_MESSAGE)
                
                # Reset temp variables
                currentScore = []
                temp_species_array =[]
                
            if WRITE_VIDEO_OUTPUT_FILE:
                out.write(image)
            if show:
                # cv2.imshow('output', image)
                self.image = image
                self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
                self.image_frame_hideConfig.setPixmap(QtGui.QPixmap.fromImage(self.image))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
                    
            if frameCount % frameThreshold == 0:
                approval_species_array = []
                species_frame_count = 0

        cv2.destroyAllWindows()
        self.image_frame_hideConfig.clear()
        self.image_frame.clear()

def ProcessRealTime_GUI(self, Yolo, output_path, input_size=YOLO_INPUT_SIZE, show=False, CLASSES=YOLO_COCO_CLASSES, score_threshold=0.6, iou_threshold=0.4, rectangle_colors=''):
    times, times_2 = [], []
    vid = cv2.VideoCapture(1)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    
    if WRITE_VIDEO_OUTPUT_FILE:
        out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    
    if WRITE_VIDEO_TRAIN_FILE:
        train = cv2.VideoWriter(str(output_path[:-4] + '-Train.mp4'), codec, fps, (width, height)) # output_path must be .mp4
        
    if WRITE_VIDEO_GUESS_FILE:
        guess = cv2.VideoWriter(str(output_path[:-4] + '-Guess.mp4'), codec, fps, (width, height)) # output_path must be .mp4
    
    frameCount = 0
    videoRecordIncrementor = 0
    min_frames_thresh = 60
    frameThreshold = 90
    species_frame_count = 0
    seen_species_array = []
    wrapped_seen_species = ''
    approval_species_array = []
    temp_species_array = []
    currentScore = []
    timer_species_array = []
    
    t = time.localtime()
    
    bbox_thick = int(0.6 * (height + width) / 1000)
    
    if bbox_thick < 1:
        bbox_thick = 1
    
    fontScale = 0.75 * bbox_thick
    
    NUM_CLASS = read_class_names(CLASSES)
    
    _, img2 = vid.read()
    
    blk = np.zeros(img2.shape, np.uint8)
    
    cv2.rectangle(blk, (20, height-20), (125, height-50), (255, 255, 255), cv2.FILLED)
    
    cv2.rectangle(blk, (width-225, height-20), (width-20, height-50), (255, 255, 255), cv2.FILLED)

    while True:
        _, img = vid.read()
        
        guess_copy = img
        currentSpecies = ''
        
        t = time.localtime()
        current_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
        
        if img is None:
            if WRITE_VIDEO_OUTPUT_FILE:
                out.release()
            if WRITE_VIDEO_TRAIN_FILE:
                train.release()
            if WRITE_VIDEO_GUESS_FILE:
                guess.release()
                os.rename(output_path[:-4] + '-Guess.mp4', 'YOLO_Videos/' + max(seen_species_array) + '-Guess' + str(videoRecordIncrementor) + '.mp4')
            cv2.destroyAllWindows()
            print("END OF VIDEO")
            break
        
        # blk = np.zeros(img.shape, np.uint8)
        
        # cv2.rectangle(blk, (20, height-20), (160, height-50), (255, 255, 255), cv2.FILLED)
        
        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = []
            for key, value in result.items():
                value = value.numpy()
                pred_bbox.append(value)
        
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')        
        
        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        
        for i, bbox in enumerate(bboxes):
            
            currentSpecies = NUM_CLASS[int(bbox[5])]
            currentScore.append(float(bbox[4]))

            approval_species_array.append(currentSpecies)
            
            if len(temp_species_array) <= i or (approval_species_array.count(currentSpecies) >= min_frames_thresh and currentSpecies not in temp_species_array):
                temp_species_array.append(currentSpecies)
            
            if currentSpecies in approval_species_array:
                
                if videoRecordIncrementor < frameCount:
                    start_time = time.strftime("%m-%d-%Y %H:%M:%S", t)
                
                species_frame_count = approval_species_array.count(currentSpecies)
                lastMaxSpecies = max(approval_species_array)
                videoRecordIncrementor = frameCount + 60
                
                if species_frame_count >= min_frames_thresh and currentSpecies not in seen_species_array:

                    seen_species_array.append(currentSpecies)
                    wrapped_seen_species = textwrap.wrap('Today\'s Seen Species: '+str(seen_species_array), width=width/20)
                    
                    print (currentSpecies + ' added to today\'s seen species')
        
        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

            gap = textsize[1] + 16

            y = 45 + i * gap
            x = 40 + textsize[0]
            
            cv2.rectangle(blk, (20, 20), (x, y+10), (255, 255, 255), cv2.FILLED)    
        
        
        # Create opacity overlay        
        image = cv2.addWeighted(image, 1.0, blk, 0.40, 1)
        
        for i, line in enumerate(wrapped_seen_species):
                        
            textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]

            gap = textsize[1] + 16

            y = 45 + i * gap
            x = int((image.shape[1] - textsize[0]) / 2)
        
            cv2.putText(image, line, (30, y-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0,0,0), bbox_thick, lineType=cv2.LINE_AA)
        
        t3 = time.time()
        times.append(t2-t1)
        times_2.append(t3-t1)
        
        times = times[-60:]
        times_2 = times_2[-60:]

        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
        frameCount += 1
        
        # Put FPS on screen
        image = cv2.putText(image, "FPS: {:.1f}".format(fps2), (30, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        
        # Put Time on screen
        image = cv2.putText(image, current_time, (width-215, height-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, (0, 0, 0), bbox_thick, lineType=cv2.LINE_AA)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        
        # print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        
        # The conditional to write a train file which is anytime the object detector doesn't see anything
        if not currentSpecies and WRITE_VIDEO_TRAIN_FILE:
            train.write(image)
        
        # The conditionals to write guess videos which is when the object detector sees something plus 2 seconds
        if videoRecordIncrementor > frameCount and WRITE_VIDEO_GUESS_FILE:
            guess.write(guess_copy)
        elif videoRecordIncrementor == frameCount and WRITE_VIDEO_GUESS_FILE:
            guess.release()
            timeWithoutDate = str(current_time[-11:])
            timeWithoutDate = timeWithoutDate.replace(':','-')
            os.rename(output_path[:-4] + '-Guess.mp4', 'YOLO_Videos/' + lastMaxSpecies + '-Guess' + '-' + str(current_time[:-9]) + '-' + timeWithoutDate + '.mp4')
            guess = cv2.VideoWriter(str(output_path[:-4] + '-Guess.mp4'), codec, fps, (width, height))
            TWITCH_SEEN_MESSAGE = 'BirdBot is ' + str("{:.2f}".format(mean(currentScore) * 100)) + '% confident it saw ' + str(temp_species_array) + ' from ' + str(start_time) + ' to ' + str(current_time)
            connection.send('BirdBotML', TWITCH_SEEN_MESSAGE)
            
            # Reset temp variables
            currentScore = []
            temp_species_array =[]
            
        if WRITE_VIDEO_OUTPUT_FILE:
            out.write(image)
        if show:
            # cv2.imshow('output', image)
            self.image = image
            self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
            self.image_frame_hideConfig.setPixmap(QtGui.QPixmap.fromImage(self.image))
            if cv2.waitKey(1) & key == ord('q'):
                cv2.destroyAllWindows()
                self.image_frame_hideConfig.clear()
                self.image_frame.clear()
                break
                
        if frameCount % frameThreshold == 0:
            approval_species_array = []
            species_frame_count = 0

    cv2.destroyAllWindows()
    self.image_frame_hideConfig.clear()
    self.image_frame.clear()
    
# Generate data from a machine learning model
def generate_ml_data_GUI(self, Yolo, video_path, input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, score_threshold=0.4, iou_threshold=0.3, rectangle_colors=''):

    print(video_path)
    
    source = "B:/BirdBot/BirdModelTraining/train_images"

    videoFiles = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.mp4)$', f)]

    print(videoFiles)

    now = datetime.now()
    current_time = now.strftime("%m-%d-%M")

    currentframe = 0
    countframes = 60
    total_count = 0
    currentSpecies = ''
    currentSpeciesArray = []
    r1 = random.randint(1, 10)

    countframes += r1

    for file in videoFiles:
        
        videoName = file
        print(videoName)
        vid = cv2.VideoCapture('B:/BirdBot/BirdModelTraining/train_images/'+videoName)
        
        writeFileName = videoName[:-4]
        
        # by default VideoCapture returns float instead of int
        imW = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        imH = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        bbox_thick = int(0.6 * (imH + imW) / 1000)
    
        if bbox_thick < 1:
            bbox_thick = 1
    
        fontScale = 0.6 * bbox_thick
        
        NUM_CLASS = read_class_names(CLASSES)
        
        while vid.isOpened():
            
            ret, frame = vid.read()
            trainframe = frame
            object_count = 0
            currentframe += 1
            
            if currentframe % countframes == 0:
                if ret:
                    if frame is None:
                        cv2.destroyAllWindows()
                        # print("END OF VIDEO")
                        break
                    
                    try:
                        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    except:
                        break
                    
                    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
                    image_data = image_data[np.newaxis, ...].astype(np.float32)

                    if YOLO_FRAMEWORK == "tf":
                        pred_bbox = Yolo.predict(image_data)
                    elif YOLO_FRAMEWORK == "trt":
                        batched_input = tf.constant(image_data)
                        result = Yolo(batched_input)
                        pred_bbox = []
                        for key, value in result.items():
                            value = value.numpy()
                            pred_bbox.append(value)
                    
                    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                    pred_bbox = tf.concat(pred_bbox, axis=0)

                    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
                    bboxes = nms(bboxes, iou_threshold, method='nms')
                    
                    try:
                        bboxes[0][0] = bboxes[0][0] - 10
                        bboxes[0][1] = bboxes[0][1]
                        bboxes[0][2] = bboxes[0][2]
                        bboxes[0][3] = bboxes[0][3]
                    except:
                        print("No detection")
                    
                    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
                    
                    objectHolder = {}
                    
                    root = ET.Element("annotation")
                    
                    # Write training photo every 5 frames on successful ID
                        
                    folder = ET.SubElement(root, "folder")
                    folder.text = "images"
                    filename = ET.SubElement(root, "filename") 
                    filename.text = writeFileName + "-" + str(currentframe) + ".jpg"
                    path = ET.SubElement(root, "path")  
                    path.text = "images/"+writeFileName + "-" + str(currentframe) + ".jpg"
                    source = ET.SubElement(root, "source")
                    
                    database = ET.SubElement(source, "database")
                    database.text = "Unspecified"
                    
                    size = ET.SubElement(root, "size")
                    
                    width = ET.SubElement(size, "width")
                    width.text = str(imW)
                    height = ET.SubElement(size, "height")
                    height.text = str(imH)
                    depth = ET.SubElement(size, "depth")
                    depth.text = str(3)
                    
                    # print(str(len(bboxes)))
                    
                    for i, bbox in enumerate(bboxes):
                        
                        # print(bbox)
                        
                        coor = np.array(bbox[:4], dtype=np.int32)
                        
                        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
                        
                        # print(coor)
                        
                        currentSpecies = NUM_CLASS[int(bbox[5])]
                        
                        currentSpeciesArray.append(currentSpecies)
                        
                        object_name = currentSpecies
                        
                        ymin = y1
                        # print(ymin)
                        xmin = x1
                        # print(xmin)
                        ymax = y2
                        # print(ymax)
                        xmax = x2
                        # print(xmax)
                        
                        middleX = (xmin + xmax) / 2
                        
                        if currentSpecies == 'American Goldfinch':
                            rectangle_colors = (150,255,255)
                        elif currentSpecies == 'American Robin':
                            rectangle_colors = (54,101,163)
                        elif currentSpecies == 'Black-capped Chickadee':
                            rectangle_colors = (23,29,29)
                        elif currentSpecies == 'Black-headed Grosbeak':
                            rectangle_colors = (0,106,188)
                        elif currentSpecies == 'Black-headed Grosbeak - Female':
                            rectangle_colors = (50,150,220)
                        elif currentSpecies == 'Chestnut-backed Chickadee':
                            rectangle_colors = (24,70,87)
                        elif currentSpecies == 'Dark-eyed Junco':
                            rectangle_colors = (57,47,45)
                        elif currentSpecies == 'European Starling':
                            rectangle_colors = (96,96,96)
                        elif currentSpecies == 'Golden-crowned Sparrow':
                            rectangle_colors = (0,205,205)
                        elif currentSpecies == 'House Finch - Male':
                            rectangle_colors = (212,170,255)
                        elif currentSpecies == 'House Finch - Female':
                            rectangle_colors = (126,139,155)
                        elif currentSpecies == 'House Sparrow':
                            rectangle_colors = (10,66,89)
                        elif currentSpecies == 'Northern Flicker':
                            rectangle_colors = (138,144,177)
                        elif currentSpecies == 'Pine Siskin':
                            rectangle_colors = (137,110,97)
                        elif currentSpecies == 'Red-breasted Nuthatch':
                            rectangle_colors = (58,95,196)
                        elif currentSpecies == 'Steller\'s Jay':
                            rectangle_colors = (191,95,0)
                        elif currentSpecies == 'Spotted Towhee':
                            rectangle_colors = (47,38,29)
                        elif currentSpecies == 'Song Sparrow':
                            rectangle_colors = (10,255,0)
                        else:
                            rectangle_colors = (255,255,255)
                            
                        if sum(rectangle_colors) >= 300:
                            fontColors = (0,0,0)
                        else: 
                            fontColors = (255,255,255)
                        
                        textsize = cv2.getTextSize(currentSpecies, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, bbox_thick)[0]
                        
                        # trainframe = cv2.putText(trainframe, str(currentSpecies), (int(middleX) - (int(textsize[0]/2)), int(ymin - 100)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, rectangle_colors, 2, lineType=cv2.LINE_AA)
                        
                        # trainframe = cv2.putText(trainframe, str(currentSpecies), (int(middleX) - (int(textsize[0]/2)), int(ymin - 100)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, fontColors, bbox_thick, lineType=cv2.LINE_AA)
                        
                        # trainframe = cv2.putText(trainframe, "V", (int(middleX), int(ymin - 75)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, rectangle_colors, 2, lineType=cv2.LINE_AA)
                        
                        # trainframe = cv2.putText(trainframe, "V", (int(middleX), int(ymin - 75)), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, fontColors, bbox_thick, lineType=cv2.LINE_AA)
                        
                        print(currentSpecies + " - Count: " + str(total_count) + " - Frame: " + str(currentframe))
                        
                        objectHolder["objectXml"+str(i)] = ET.SubElement(root, "object")
                            
                        objectHolder["nameXml"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "name")
                        objectHolder["nameXml"+str(i)].text = object_name
                        objectHolder["pose"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "pose")
                        objectHolder["pose"+str(i)].text = "Unspecified"
                        objectHolder["truncated"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "truncated") 
                        objectHolder["truncated"+str(i)].text = "Unspecified"
                        objectHolder["difficult"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "difficult")
                        objectHolder["difficult"+str(i)].text = "Unspecified"
                        
                        objectHolder["bndbox"+str(i)] = ET.SubElement(objectHolder["objectXml"+str(i)], "bndbox") 
                        
                        objectHolder["xminXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "xmin")
                        objectHolder["xminXml"+str(i)].text = str(xmin)
                        objectHolder["yminXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "ymin") 
                        objectHolder["yminXml"+str(i)].text = str(ymin)
                        objectHolder["xmaxXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "xmax") 
                        objectHolder["xmaxXml"+str(i)].text = str(xmax)
                        objectHolder["ymaxXml"+str(i)] = ET.SubElement(objectHolder["bndbox"+str(i)], "ymax") 
                        objectHolder["ymaxXml"+str(i)].text = str(ymax)
                        # print(objectHolder["ymaxXml"+str(i)].text)
                        
                        # print (ET.tostring(root))
                        
                        object_count += 1
                        total_count += 1
                        
                    if object_count > 0:
                        
                        name = "B:/BirdBot/BirdModelTraining/train_images/" + writeFileName + "-" + str(currentframe) + ".jpg"
                        
                        nameCheck = "B:/BirdBot/BirdModelTraining/train_images/" + writeFileName + "-" + str(currentframe) + "-Check" + ".jpg"
                        
                        cv2.imwrite(name, trainframe)
                        
                        cv2.imwrite(nameCheck, image)
                        
                        # print("Wrote: " + name)
                        
                        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
                        
                        xmlName = 'B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml'
                        
                        with open('B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml', "w") as f:
                            
                            f.write(xmlstr)
                            
                            # print("Wrote: " + xmlName)
                            
                        a_file = open('B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml', "r")
                            
                        lines = a_file.readlines()
                        a_file.close()
                        
                        del lines[0]
                        
                        new_file = open('B:/BirdBot/BirdModelTraining/train_images/' + writeFileName + "-" + str(currentframe) + '.xml', "w+")
                        
                        for line in lines:
                            new_file.write(line)
                        
                        new_file.close()
                        
                        object_count = 0
                        currentSpeciesArray = []
                    
                else:
                    vid.release()
                    currentframe = 0
                    cv2.destroyAllWindows()
                    break
            
            elif frame is None:
                currentframe = 0
                cv2.destroyAllWindows()
                # print("END OF VIDEO")
                break
                    
        if file is None:
            cv2.destroyAllWindows()
            # print("END OF VIDEO")
            break