import os
import gc
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
import tqdm
from PIL import Image

from retinanet import build_head, get_backbone, convert_to_corners, visualize_detections, resize_and_pad_image, prepare_image
from retinanet import FeaturePyramid, RetinaNet, AnchorBox, DecodePredictions
from efficientnet import EfficientNet


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
# resnet50_backbone = get_backbone()
# model = RetinaNet(num_classes, resnet50_backbone)
# efficientnet = EfficientNet(num_classes, 224)
# print(efficientnet.call)

import tensorflow_hub as hub
from tensorflow.keras import layers
feature_extractor_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"

# feature_extractor_layer = hub.KerasLayer(
#     feature_extractor_url,
#     input_shape=(224,224,3))
# # feature_extractor_layer.trainable = False
# label_model = tf.keras.Sequential([
#   feature_extractor_layer,
#   layers.Dense(num_classes, activation='softmax', name="class_label")
# ])


# Change this to `model_dir` when not using the downloaded weights

# label_weights_dir = "./efficientnet"


# latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
# model.load_weights(latest_checkpoint)
# latest_label_checkpoint = tf.train.latest_checkpoint(label_weights_dir)
# label_model.load_weights(latest_label_checkpoint)

#### efficient net ####
# image = tf.keras.Input(shape=[None, None, 3], name="image")
# label_detections = label_model(image, training=False)
# label_inference_model = tf.keras.Model(inputs=image, outputs=label_detections)


#### retinanet ###


# image = tf.keras.Input(shape=[None, None, 3], name="image")
# predictions = model(image, training=False)
# detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
# label_inference_model = tf.keras.Model(inputs=image, outputs=label_detections)

# load validation image
# val_dataset = tf.data.TFRecordDataset("train03-1267.tfrecords") \
#     .map(parse_example) \


###                        ###
### Show inferenced images ###
###                        ###
# int2str = (
#     'Typical Appearance',
#     'Indeterminate Appearance',
#     'Atypical Appearance',
#     'Negatie for Pneumonia')

# for sample in val_dataset.take(5):
#     # print(sample[0])
#     image = sample[0]
#     gclasses = sample[1][0]
#     gboxes = sample[1][1]
    
#     input_image, ratio = prepare_image(image)
#     # print(input_image.shape)
#     detections = inference_model.predict(input_image)
#     num_detections = detections.valid_detections[0]
#     # print(detections.nmsed_classes[0][:num_detections])
#     class_names = [
#         int2str[int(x)] for x in detections.nmsed_classes[0][:num_detections]
#     ]
#     # print(class_names)
    
#     visualize_detections(
#         image,
#         detections.nmsed_boxes[0][:num_detections] / ratio, 
#         class_names,
#         detections.nmsed_scores[0][:num_detections],
#         gboxes,
#         int2str[int(gclasses)],
#     )



def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dic = dicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dic.pixel_array, dic)
    else:
        data = dic.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dic.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

def resize_xray(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im


# submission format
int2str = (
    'typical',
    'indeterminate',
    'atypical',
    'negative')

sub_df = pd.read_csv(f'./sample_submission.csv')
study_df = sub_df.loc[sub_df.id.str.contains('_study')]
image_df = sub_df.loc[sub_df.id.str.contains('_image')]
IMG_SIZE = 512
TEST_PATH = f'./tmp/test/'



###
### convert test file and get info ###
###
# def prepare_test_images():
#     image_id = []
#     dim0 = []
#     dim1 = []

#     os.makedirs(TEST_PATH, exist_ok=True)

#     for dirname, _, filenames in tqdm.tqdm(os.walk(f'./test')):
#         for filename in filenames:
#             # set keep_ratio=True to have original aspect ratio
#             xray = read_xray(os.path.join(dirname, filename))
#             im = resize_xray(xray, size=IMG_SIZE)  
#             im.save(os.path.join(TEST_PATH, filename.replace('dcm', 'png')))

#             image_id.append(filename.replace('.dcm', ''))
#             dim0.append(xray.shape[0])
#             dim1.append(xray.shape[1])
            
#     return image_id, dim0, dim1

# image_ids, dim0, dim1 = prepare_test_images()
# print(f'Number of test images: {len(os.listdir(TEST_PATH))}')



# meta_df = pd.DataFrame.from_dict( \
#     {'image_id': image_ids, 'dim0': dim0, 'dim1': dim1})

# Associate image-level id with study-level ids.
# Note that a study-level might have more than one image-level ids.
# for study_dir in os.listdir('./test'):
#     for series in os.listdir(f'./test/{study_dir}'):
#         for image in os.listdir(f'./test/{study_dir}/{series}/'):
#             image_id = image[:-4]
#             meta_df.loc[meta_df['image_id'] == image_id, 'study_id'] = study_dir
# meta_df.to_csv("meta_df.csv")




###
### inference image level ####
###
# image_df['path'] = \
#     image_df.apply(lambda row: TEST_PATH+row.id.split('_')[0]+'.png', axis=1)
# image_df = image_df.reset_index(drop=True)


# predictions = []
# for sample in image_df['path']:
#     print(sample)

#     tmp = []
#     # without batch
#     image = tf.keras.preprocessing.image.load_img( \
#         sample, color_mode='rgb', target_size=(224,224))
#     image = tf.keras.preprocessing.image.img_to_array(image)
#     input_image, ratio = prepare_image(image)
#     preds = label_inference_model.predict(input_image)

#     # print(preds)
#     tmp.extend(preds[0])
    
# predictions.append(tmp)
# predictions = np.mean(predictions, axis=0)

# class_labels = ['0', '1', '2', '3']
# image_df.loc[:, class_labels] = predictions
# # print(image_df.head())


# meta_df =  pd.read_csv("meta_df.csv")
# PRED_PATH = './tmp/labels'
# prediction_files = os.listdir(PRED_PATH)
# print(f'Number of opacity predicted: {len(prediction_files)}')


### get label info 
# class_to_id = { 
#     'negative': 0,
#     'typical': 1,
#     'indeterminate': 2,
#     'atypical': 3}
# id_to_class  = {v:k for k, v in class_to_id.items()}

# def get_study_prediction_string(preds, threshold=0):
#     string = ''
#     for idx in range(4):
#         conf =  preds[idx]
#         if conf>threshold:
#             string+=f'{id_to_class[idx]} {conf:0.2f} 0 0 1 1 '
#     string = string.strip()
#     return string


### extrac label info from image_df into study_df 
# study_ids = []
# pred_strings = []
# for study_id, df in meta_df.groupby('study_id'):
#     # accumulate preds for diff images belonging to same study_id
#     tmp_pred = []
    
#     df = df.reset_index(drop=True)
#     for image_id in df.image_id.values:
#         preds = image_df.loc[image_df.id == image_id+'_image'].values[0]
#         tmp_pred.append(preds[3:])
    
#     preds = np.mean(tmp_pred, axis=0)
#     pred_string = get_study_prediction_string(preds)
#     pred_strings.append(pred_string)
    
#     study_ids.append(f'{study_id}_study')
    
# study_df = pd.DataFrame.from_dict( \
#     {'id': study_ids, 'PredictionString': pred_strings})
# print(study_df.head())


####                       ####
#### Inference Image Label ####
####                       ####

# image = tf.keras.Input(shape=[None, None, 3], name="image")
# predictions = model(image, training=False)
# detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
# inference_model = tf.keras.Model(inputs=image, outputs=detections)

image_model_path = "retinanet/saved"
inference_model = keras.models.load_model(image_model_path, compile=False)
# print(predictions)


# image = tf.keras.Input(shape=[None, None, 3], name="image")
# inference_model = DecodePredictions(confidence_threshold=0.5)(image, predictions)
# inference_model = tf.keras.Model(inputs=image, outputs=detections)(detectinos)



def _decode_box_predictions(anchor_boxes, box_predictions):
    _box_variance=[0.1, 0.1, 0.2, 0.2]
    boxes = box_predictions * _box_variance
    boxes = tf.concat(
        [
            boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
            tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
        ],
        axis=-1,
    )
    boxes_transformed = convert_to_corners(boxes)
    return boxes_transformed

class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

_anchor_box = AnchorBox()



PRED_PATH = './tmp/boxes'
os.makedirs(PRED_PATH, exist_ok=True)
for sample in os.listdir('./tmp/test'):
    # print(sample)
    path = f'./tmp/test/{sample}'
    # print(path)
    image = tf.keras.preprocessing.image.load_img( \
        path, color_mode='rgb', target_size=(512,512))
    image = tf.keras.preprocessing.image.img_to_array(image)
    input_image, ratio = prepare_image(image)


    predictions = inference_model(input_image)
    # print(predictions)

    image_shape = tf.cast(tf.shape(input_image), dtype=tf.float32)
    anchor_boxes = _anchor_box.get_anchors(image_shape[1], image_shape[2])

    box_predictions = predictions[:,:,:4]
    cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
    boxes = _decode_box_predictions(anchor_boxes[None, ...], box_predictions)

    nmsed_boxes, nmsed_scored, nmsed_classes, num_detections = \
        tf.image.combined_non_max_suppression(
        tf.expand_dims(boxes, axis=2),
        cls_predictions,
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.05,
        clip_boxes=False,
        )


    class_names = [
        int2str[int(x)] for x in nmsed_classes.numpy()[0][:int(num_detections)]
    ]
    # print(class_names)
    

    filename = sample[:-4]
    print(f'{PRED_PATH}/{filename}.txt')
    with open(f'{PRED_PATH}/{filename}.txt',"w") as txtfile:
        out_array = []
        for label, box in zip(nmsed_classes[0][:int(num_detections)], nmsed_boxes[0][:int(num_detections)]):        
            out_array = tf.concat([[label], box], axis=0)
            txtfile.write(' '.join(str(item) for item in out_array.numpy()) + '\n') 





# def scale_bboxes_to_original(row, bboxes):
#     # Get scaling factor
#     scale_x = IMG_SIZE/row.dim1
#     scale_y = IMG_SIZE/row.dim0
    
#     scaled_bboxes = []
#     # print("decoded:")
#     # print(bboxes)
#     for bbox in bboxes:
#         xmin, ymin, xmax, ymax = bbox
        
#         xmin = int(np.round(xmin/scale_x))
#         ymin = int(np.round(ymin/scale_y))
#         xmax = int(np.round(xmax/scale_x))
#         ymax = int(np.round(ymax/scale_y))
        
#         scaled_bboxes.append([xmin, ymin, xmax, ymax])
#     # print("scaled:")
#     # print(scaled_bboxes)

# # Read the txt file generated by YOLOv5 during inference and extract 
# # confidence and bounding box coordinates.
# def get_conf_bboxes(file_path):
#     confidence = []
#     bboxes = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             preds = line.strip('\n').split(' ')
#             preds = list(map(float, preds))

#             # confidence.append(preds[-1])
#             confidence.append(preds[0])
#             # bboxes.append(preds[1:-1])
#             bboxes.append(preds[1:])
#     return confidence, bboxes


# #### Merge results ####
# #### Image Box ####
# # print(image_df)
# image_pred_strings = []
# for i in tqdm.tqdm(range(len(image_df))):
#     row = meta_df.loc[i]
#     id_name = row.image_id
    
#     if f'{id_name}.txt' in prediction_files:
#         # opacity label
#         confidence, bboxes = get_conf_bboxes(f'{PRED_PATH}/{id_name}.txt') 
#         ori_bboxes = scale_bboxes_to_original(row, bboxes)
        
#         pred_string = ''
#         for j, conf in enumerate(confidence):
#             pred_string += f'opacity {conf} ' + ' '.join(map(str, bboxes[j])) + ' '
#         image_pred_strings.append(pred_string[:-1]) 
#     else:
#         image_pred_strings.append("None 1 0 0 1 1")


# meta_df['PredictionString'] = image_pred_strings
# image_df = meta_df[['image_id', 'PredictionString']]
# image_df.insert(0, 'id', image_df.apply(lambda row: row.image_id+'_image', axis=1))
# image_df = image_df.drop('image_id', axis=1)
# # print(image_df.head())

# #### Write result ####
# sub_df = pd.concat([study_df, image_df])
# sub_df.to_csv('submission.csv', index=False)
# print(sub_df.head)