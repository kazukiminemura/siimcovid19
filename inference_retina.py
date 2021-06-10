import os
import re
import zipfile

import numpy as np
import tensorflow as tf
# https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pandas as pd
import pydicom as dicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import albumentations as A
import cv2
import glob
import pprint


from retinanet import build_head, get_backbone, convert_to_corners, visualize_detections, resize_and_pad_image, prepare_image
from retinanet import FeaturePyramid, RetinaNet, AnchorBox, DecodePredictions, DecodePredictions


###                ###
### INFERENCE MAIN ###
###                ###

@tf.function
def parse_example(record):
    keys_to_features = {
        "image/image": tf.io.FixedLenFeature([], tf.string),
        "image/label": tf.io.FixedLenFeature([], tf.int64),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    image = tf.io.decode_raw(parsed["image/image"], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, shape=[512,512,3])
    # image = tf.image.resize(image, [256,256])
    # image = image / 255.0
    label = tf.cast(parsed["image/label"], tf.int32)

    xmin = tf.expand_dims(parsed["image/object/bbox/xmin"].values, 0)
    ymin = tf.expand_dims(parsed["image/object/bbox/ymin"].values, 0)
    xmax = tf.expand_dims(parsed["image/object/bbox/xmax"].values, 0)
    ymax = tf.expand_dims(parsed["image/object/bbox/ymax"].values, 0)
    bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    bboxes = tf.transpose(bboxes) # 4*N -> N*4
    bboxes *= 512
    

    return image, (label, bboxes)


num_classes = 4
resnet50_backbone = get_backbone()
model = RetinaNet(num_classes, resnet50_backbone)

# Change this to `model_dir` when not using the downloaded weights
weights_dir = "retinanet"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

# load validation image
val_dataset = tf.data.TFRecordDataset("train00-1267.tfrecords") \
    .map(parse_example) \

int2str = (
    'Typical Appearance',
    'Indeterminate Appearance',
    'Atypical Appearance',
    'Negatie for Pneumonia')


for sample in val_dataset.take(5):
    # print(sample[0])
    image = sample[0]
    gclasses = sample[1][0]
    gboxes = sample[1][1]
    
    input_image, ratio = prepare_image(image)
    # print(input_image.shape)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    # print(detections.nmsed_classes[0][:num_detections])
    class_names = [
        int2str[int(x)] for x in detections.nmsed_classes[0][:num_detections]
    ]
    # print(class_names)
    
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio, 
        class_names,
        detections.nmsed_scores[0][:num_detections],
        gboxes,
        int2str[int(gclasses)],
    )


