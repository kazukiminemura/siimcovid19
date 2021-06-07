import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

# SEED=42
# name2label = {'Typical Appearance': 3,
#  'Indeterminate Appearance': 1,
#  'Atypical Appearance': 2,
#  'Negative for Pneumonia': 0}
# class_names = list(name2label.keys())
# label2name = {v:k for k, v in name2label.items()}

# def display_batch(batch, size=2):
#     imgs, tars = batch
#     plt.figure(figsize=(size*5, 5))
#     for img_idx in range(size):
#         plt.subplot(1, size, img_idx+1)
#         plt.title(f'class: {label2name[tars[img_idx].numpy()[0]]}', fontsize=15)
#         plt.imshow(imgs[img_idx,:, :, :])
#         plt.xticks([])
#         plt.yticks([])
#     plt.tight_layout()
#     plt.show()


# import re, math
# def decode_image(image_data):
#     image = tf.image.decode_png(image_data, channels=3)
#     image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
#     image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
#     return image
# def prepare_target(target):    
#     target = tf.cast(target, tf.float32)            
#     target = tf.reshape(target, [1])         
#     return target

# def read_labeled_tfrecord(example):
#     LABELED_TFREC_FORMAT = {
#         "image" : tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
#         "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
#     }
#     example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
#     image = decode_image(example['image'])
#     image  = tf.reshape(image, [DIM, DIM, 3])
#     target = prepare_target(example['target'])
#     return image, target # returns a dataset of (image, label) pairs

# def load_dataset(fileids, labeled=True, ordered=False):
#     # Read from TFRecords. For optimal performance, reading from multiple files at once and
#     # disregarding data order. Order does not matter since we will be shuffling the data anyway.

#     ignore_order = tf.data.Options()
#     if not ordered:
#         ignore_order.experimental_deterministic = False # disable order, increase speed

#     dataset = tf.data.TFRecordDataset(fileids, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
#     dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
#     dataset = dataset.map(read_labeled_tfrecord)
#     # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
#     return dataset

# def get_training_dataset():
#     dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
#     dataset = dataset.repeat() # the training dataset must repeat for several epochs
#     dataset = dataset.shuffle(20, seed=SEED)
#     dataset = dataset.batch(BATCH_SIZE)
#     dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
#     return dataset

# def count_data_items(fileids):
#     # the number of data items is written in the id of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
#     n = [int(re.compile(r"-([0-9]*)\.").search(fileid).group(1)) for fileid in fileids]
#     return np.sum(n)

# ### INITIALIZE VARIABLES ###
# DIM=512
# IMAGE_SIZE= [DIM,DIM]
# BATCH_SIZE = 32
# AUTO = tf.data.experimental.AUTOTUNE
# TRAINING_FILENAMES = tf.io.gfile.glob('train*.tfrec')
# TEST_FILENAMES     = tf.io.gfile.glob('test*.tfrec')
# print('There are %i train & %i test images'%(\
#     count_data_items(TRAINING_FILENAMES), count_data_items(TEST_FILENAMES)))


# ### DISPLAY TRAIN IMAGES ###
# training_dataset = get_training_dataset()
# training_dataset = training_dataset.unbatch().batch(20)
# train_batch = next(iter(training_dataset))
# display_batch(train_batch, 5)




