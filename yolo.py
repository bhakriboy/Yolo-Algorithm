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
import yolo_filter_boxes
import iou
import yolo_non_max_suppression
import yolo_eval
import predict

sess = K.get_session()

print('yes')


class_names = read_classes("C:/Users/Vikarn Bhakri/Desktop/model_data/coco_classes.txt")
anchors = read_anchors("C:/Users/Vikarn Bhakri/Desktop/model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    

print('yes1')


yolo_model = load_model("C:/Users/Vikarn Bhakri/Desktop/model_data/yolo.h5")

print('yes2')


yolo_model.summary()

print('yes1')

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

print('yes1')


scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
print('yes1')

out_scores, out_boxes, out_classes = predict(sess, "download.jpg")
print (scores)
print('yes1')
