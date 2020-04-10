import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
    
    
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs

    
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs)
    
    
    boxes = scale_boxes(boxes, image_shape)

    
    scores, boxes, classes = yolo_non_max_suppression(scores,boxes,classes)
    
    
    
    return scores, boxes, classes
