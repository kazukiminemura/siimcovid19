# Built In Imports
from glob import glob
from pandarallel import pandarallel
from random import shuffle
from tqdm import tqdm
import ast
import cv2
import numpy as np
import os
import pandas as pd
import plotly.express as px
import pprint
import pydicom
import random
import seaborn as sns
import sys
import tensorflow as tf


print("\n... SEEDING FOR DETERMINISTIC BEHAVIOUR ...")
def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_it_all()
print("... SEEDING COMPLETE ...\n\n")


print("\n... SETTING PRESETS STARTING...")
FIG_FONT = dict(family="Helvetica, Arial", size=14, color="#7f7f7f")
LABEL_LIST = ['atypical', 'indeterminate', 'negative', 'typical']
LABEL_COLORS = [px.colors.label_rgb(px.colors.convert_to_RGB_255(x)) for x in sns.color_palette("Spectral", 4)]
# LABEL_COLORS_WOUT_NO_FINDING = LABEL_COLORS[:8]+LABEL_COLORS[9:]
print("... SETTING PRESETS COMPLETE...\n\n")


#### preprocess functions ####
def unpack_bbox_column(df_row):
    """ go from xmin,ymin,width,height --> xmin,ymin,xmax,ymax """
    df_row["xmin"] = df_row["boxes"]["x"] if df_row["boxes"]["x"] > 0 else 1.0
    df_row["ymin"] = df_row["boxes"]["y"] if df_row["boxes"]["y"] > 0 else 1.0
    df_row["xmax"] = df_row["boxes"]["x"]+df_row["boxes"]["width"]
    df_row["ymax"] = df_row["boxes"]["y"]+df_row["boxes"]["height"]
    return df_row

def get_human_label(row):
    """ Get human readable label for visualization purposes """
    for lbl in ["negative", "typical", "indeterminate", "atypical"]:
        if row[lbl]:
            row["human_label"] = lbl
    return row

def get_img_shape(path):
    """ Return (width, height) """
    dcm = pydicom.read_file(path)
    return dcm.Rows, dcm.Columns

def create_fractional_bbox_coordinates(row):
    """ Function to return bbox coordiantes as fractions from DF row """
    row["frac_xmin"] = row["xmin"]/row["width"]

# c = list(zip(train_df.dcm_path, train_df.human_label))
# shuffle(c)
# addrs, labels = zip(*c)


def get_absolute_file_paths(directory):
    all_abs_file_paths = []
    for dirpath,_,filenames in tqdm(os.walk(directory)):
        for f in filenames:
            all_abs_file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return all_abs_file_paths



# Define the root data directory
ROOT_DIR = "/home/cvalgo/projects"
DATA_DIR = os.path.join(ROOT_DIR,"covid")

# Define the paths to the training and testing dicom folders respectively
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Define paths to the relevant csv files
TRAIN_IMAGE_LVL_CSV = os.path.join(DATA_DIR, "train_image_level.csv")
TRAIN_STUDY_LVL_CSV = os.path.join(DATA_DIR, "train_study_level.csv")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

# Create the relevant dataframe objects
print("\n... OPEN THE IMAGE LEVEL DATAFRAME ...\n")
train_image_level_df = \
    pd.read_csv(TRAIN_IMAGE_LVL_CSV, usecols=["id", "boxes", "StudyInstanceUID"]).sort_values(by="StudyInstanceUID").reset_index(drop=True)

print("\n... CONVERT BOUNDING BOX STRING REPRESENTATION TO LITERAL ...\n")
train_image_level_df["boxes"] = train_image_level_df["boxes"].fillna('{"x":0, "y":0, "width":1, "height":1}')
train_image_level_df["boxes"] = train_image_level_df["boxes"].apply(lambda x: ast.literal_eval(x))

print("\n... OPEN THE STUDY LEVEL DATAFRAME ...\n")
train_study_level_df = pd.read_csv(TRAIN_STUDY_LVL_CSV)

print("\n... RENAME AND FIX SOME COLUMNS IN BOTH DATAFRAMES ...\n")
train_study_level_df.columns = ["StudyInstanceUID", "negative", "typical", "indeterminate", "atypical"]
train_study_level_df["StudyInstanceUID"] = train_study_level_df["StudyInstanceUID"].str.replace("_study", "")
train_image_level_df["id"] = train_image_level_df["id"].str.replace("_image", "")

print("\n... OPEN THE SUBMISSION DATAFRAME ...\n")
ss_df = pd.read_csv(SS_CSV)

print("\n... GET THE TEST DATASET PATHS ...\n")
all_dcm_paths_test = get_absolute_file_paths(TEST_DIR)
ss_df["dcm_path"] = ss_df["id"].str.replace("_study", "")\
                               .str.replace("_image", "")\
                               .apply(lambda x: [z for z in all_dcm_paths_test if x in z][-1])


print("\n... COMBINE IMAGE AND STUDY LEVEL DATAFRAMES ...\n")
train_df = pd.merge(train_image_level_df, train_study_level_df, on="StudyInstanceUID")

print("\n... EXPLODE LIST OF BOUNDING BOXES INTO INDIVIDUAL ROWS ...\n")
train_df["boxes"] = train_df.boxes.apply(lambda x: [x] if type(x)==dict else x)
train_df["boxes"] = train_df.boxes
train_df = train_df.explode("boxes", ignore_index=True).reset_index(drop=True)

print("\n... CONVERT BBOX COORDINATES INTO RESPECTIVE COLUMNS ...\n")
train_df = train_df.apply(unpack_bbox_column, axis=1)
train_df = train_df.sort_values(by="id").reset_index(drop=True)

print("\n... CREATE SERIES INSTANCE UID COLUMN...\n")
train_df["SeriesInstanceUID"] = train_df.StudyInstanceUID.apply(lambda x: os.listdir(os.path.join(TRAIN_DIR, x))[0])

print("\n... CREATE IMAGE INSTANCE UID COLUMN...\n")
train_df["ImageInstanceUID"] = train_df.apply(lambda row: (os.listdir(os.path.join(TRAIN_DIR, row.StudyInstanceUID, row.SeriesInstanceUID))[0][:-4]), axis=1)

print("\n... CREATE PATH TO DICOM FILE COLUMN...\n")
all_dcm_paths_train = get_absolute_file_paths(TRAIN_DIR)
train_df["dcm_path"] = train_df["id"].apply(lambda x: [z for z in all_dcm_paths_train if x in z][-1])

print("\n... CREATE HUMAN LABEL COLUMN ...\n")
train_df = train_df.apply(get_human_label, axis=1)

print("\n... REORDER TRAIN DATAFRAME ...\n")
train_df = train_df[["id", "dcm_path", "xmin", "ymin", "xmax", "ymax", "human_label", "negative", "typical", "indeterminate", "atypical", "StudyInstanceUID", "SeriesInstanceUID", "ImageInstanceUID"]]

# print(train_df.human_label)
print("\n... CREATE INTEGER LABEL COLUMN ...\n")
train_df["integer_label"] = train_df["human_label"].apply(lambda x: LABEL_LIST.index(x))


print("\n... THIS STEP WILL TAKE 5-10 MINUTES ...\n")
# Initialization
pandarallel.initialize()
train_df["img_shape"] = train_df.dcm_path.parallel_apply(lambda x: get_img_shape(x))
train_df["width"] = train_df.img_shape.apply(lambda x: x[1])
train_df["height"] = train_df.img_shape.apply(lambda x: x[0])

ss_df["img_shape"] = ss_df.dcm_path.parallel_apply(lambda x: get_img_shape(x))
ss_df["width"] = ss_df.img_shape.apply(lambda x: x[1])
ss_df["height"] = ss_df.img_shape.apply(lambda x: x[0])

print("\n... IMAGE SHAPE DETERMINATION COMPLETE ...\n")

print("\n... CREATE FRACTIONAL BBOX COORDINATES ...\n")
train_df = train_df.apply(create_fractional_bbox_coordinates, axis=1)



print("\n\nCOMBINED AND EXPLODED TRAIN DATAFRAME\n\n")
# display(train_df)
train_df.to_csv("train_metadata.csv")

print("\n\nSAMPLE SUBMISSION DATAFRAME\n\n")
# display(ss_df)
ss_df.to_csv("submission_metadata.csv")



# #### get from https://github.com/kalaspuffar/tensorflow-data/blob/master/create_dataset.py

# # Define the root data directory
# ROOT_DIR = "/home/cvalgo/projects"
# DATA_DIR = os.path.join(ROOT_DIR,"covid")
# TRAIN_DIR = os.path.join(DATA_DIR, "train")
# train_df = pd.read_csv(os.path.join(DATA_DIR, "train_metadata.csv"))

# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# def _float_feature_list(value):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# def load_image(addr):
#     # read an image and resize to (224, 224)
#     # cv2 load images as BGR, convert it to RGB
#     img = cv2.imread(addr)
#     if img is None:
#         return None
#     img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img


# import pydicom
# from pydicom.pixel_data_handlers.util import apply_voi_lut
# def dicom2array(path, voi_lut=True, fix_monochrome=True):
#     """ Convert dicom file to numpy array 
    
#     Args:
#         path (str): Path to the dicom file to be converted
#         voi_lut (bool): Whether or not VOI LUT is available
#         fix_monochrome (bool): Whether or not to apply monochrome fix
        
#     Returns:
#         Numpy array of the respective dicom file 
        
#     """
#     # Use the pydicom library to read the dicom file
#     dicom = pydicom.read_file(path)
    
#     # VOI LUT (if available by DICOM device) is used to 
#     # transform raw DICOM data to "human-friendly" view
#     if voi_lut:
#         data = apply_voi_lut(dicom.pixel_array, dicom)
#     else:
#         data = dicom.pixel_array
        
#     # The XRAY may look inverted
#     #   - If we want to fix this we can
#     if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
#         data = np.amax(data) - data
    
#     # Normalize the image array and return
#     data = data - np.min(data)
#     data = data / np.max(data)
#     data = (data * 255).astype(np.uint8)
#     return data

# def createDataRecord(out_filename, addrs, labels):
#     # open the TFRecords file
#     writer = tf.io.TFRecordWriter(out_filename)
#     for i in range(len(addrs)):
#         # print how many images are saved every 1000 images
#         if not i % 1000:
#             print('Train data: {}/{}'.format(i, len(addrs)))
#             sys.stdout.flush()
#         # Load the image
#         img = dicom2array(addrs[i])

#         label = labels[i]

#         # bbox = bboxs[i]

#         if img is None:
#             continue

#         # Create a feature
#         feature = {
#             'image_raw': _bytes_feature(img.tostring()),
#             'label': _int64_feature(label)
#             # 'bbox' : _float_feature_list(bbox)
#         }

#         # Create an example protocol buffer
#         example = tf.train.Example(features=tf.train.Features(feature=feature))
        
#         # Serialize to string and write on the file
#         writer.write(example.SerializeToString())
        
#     writer.close()
#     sys.stdout.flush()


# # # to shuffle data
# # boxes = np.vstack(
# #     (np.array(train_df.xmin.tolist()),
# #     np.array(train_df.ymin.tolist()),
# #     np.array(train_df.xmax.tolist()),
# #     np.array(train_df.ymax.tolist())))
# # print(boxes.shape)

# c = list(zip(train_df.dcm_path.values.tolist(), train_df.integer_label.values.tolist()))
# shuffle(c)
# addrs, labels = zip(*c)

# # print(len(train_df.dcm_path.values.tolist()))
# # print(len(addrs))

# # Divide the data into 60% train, 20% validation, and 20% test
# train_addrs = addrs[0:int(0.6*len(addrs))]
# train_labels = labels[0:int(0.6*len(labels))]
# # train_bboxes = bboxes[0:int(0.6*len(bboxes))]
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
# # test_addrs = addrs[int(0.8*len(addrs)):]
# # test_labels = labels[int(0.8*len(labels)):]

# createDataRecord('train.tfrecords', train_addrs, train_labels)
# createDataRecord('val.tfrecords', val_addrs, val_labels)
# # createDataRecord('test.tfrecords', test_addrs, test_labels)