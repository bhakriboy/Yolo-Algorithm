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

def predict(sess,image_file ):
   

    
    image, image_data = preprocess_image("C:/Users/Vikarn Bhakri/Desktop/images/" + image_file, model_image_size = (608, 608))

    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    

    
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
   
    colors = generate_colors(class_names)
    
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
   
    image.save(os.path.join("C:/Users/Vikarn Bhakri/Desktop/out/", image_file), quality=90)

    output_image = scipy.misc.imread(os.path.join("C:/Users/Vikarn Bhakri/Desktop/out/", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes
