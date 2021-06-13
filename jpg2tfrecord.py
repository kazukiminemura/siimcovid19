import numpy as np 
import pandas as pd 
import os, shutil
import random
import matplotlib.pyplot as plt

from glob import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from tqdm.notebook import tqdm
tqdm.pandas()

## logging setup
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False


# parameters
SEED  = 42
FOLDS = 5
DIM   = 512
np.random.seed(SEED)
random.seed(SEED)

print(f'./input/siimcovid19-{DIM}-jpg-image-dataset/train.csv')
train_df = pd.read_csv( \
    f'./input/siimcovid19-{DIM}-jpg-image-dataset/train.csv')
# train_df['dcm_image_path'] = train_df['filepath']
train_df['image_path'] = \
    f'./input/siimcovid19-{DIM}-jpg-image-dataset/train/'+train_df.image_id+'.jpg'

# print(train_df.head(2))

name2label = {'Typical Appearance': 3,
 'Indeterminate Appearance': 1,
 'Atypical Appearance': 2,
 'Negative for Pneumonia': 0}
class_names = list(name2label.keys())
label2name = {v:k for k, v in name2label.items()}
train_df['class_name']  = \
    train_df.progress_apply(lambda row:row[class_names].iloc[ \
    [row[class_names].values.argmax()]].index.tolist()[0], axis=1)
train_df['class_label'] = train_df.class_name.map(name2label)

# print(train_df.head()["boxes"])
# print(train_df.head()["class_name"])

train_df["class_name"].hist()
plt.tight_layout()
plt.show()


'''
### Stratified KFold by Groups ###filepath
from sklearn.model_selection import GroupKFold, StratifiedKFold
gkf  = GroupKFold(n_splits = 5)
train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate( \
    gkf.split(train_df, groups = train_df.StudyInstanceUID.tolist())):
    train_df.loc[val_idx, 'fold'] = fold
# print(train_df.head())


# ### show sample image ###
# import matplotlib.pyplot as plt
import cv2
def load_image(path, dim=DIM, ch=3):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if ch==None else cv2.IMREAD_COLOR)
    if img.shape[:2]!=(dim,dim) and dim!=-1:
        img = cv2.resize(img, dsize=(dim,dim), interpolation=cv2.INTER_AREA)
    return img

# plt.imshow(load_image(train_df.image_path.iloc[100], dim=-1))# show 100th image
# print(load_image(train_df.image_path.iloc[100]))


# get dcm image info
import pydicom
def get_img_shape(path):
    """ Return (width, height) """
    dcm = pydicom.read_file(path)
    return dcm.Rows, dcm.Columns



### TFRcord Data ###
import tensorflow as tf

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def train_serialize_example( \
    feature0, feature1, feature2, feature3, feature4, feature5 ,feature6,feature7,feature8):
    feature = {
      'image/image'   : _bytes_feature(feature0),
      'image/image_id'   : _bytes_feature(feature1),
      'image/objects_num'   : _int64_feature(feature2),
      'image/object/bbox/xmin'    : _float_list_feature(feature3),
      'image/object/bbox/ymin'    : _float_list_feature(feature4),
      'image/object/bbox/xmax'    : _float_list_feature(feature5),
      'image/object/bbox/ymax'    : _float_list_feature(feature6),
      'image/object/bbox/label'    : _int64_list_feature(feature7),
      'image/label'   : _int64_feature(feature8),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()





import ast
show=True
folds = train_df.fold.unique().tolist()
# print(folds)
for fold in tqdm(folds): # create tfrecord for each fold
    fold_df = train_df[train_df.fold==fold]
    if show:
        print(); print('Writing TFRecord of fold %i :'%(fold))

    with tf.io.TFRecordWriter('train%.2i-%i.tfrecords'%(fold,fold_df.shape[0])) as writer:
        samples = fold_df.shape[0]
        it = tqdm(range(samples)) if show else range(samples)
        for k in it: # images in fold
            row = fold_df.iloc[k,:]
            image_raw  = load_image(row['image_path'], dim=DIM)
            image_id   = row['image_id']
            boxs      = row['boxes']
            label     = np.array(row['class_label'], dtype=np.uint8)
            # box_labels = row['label']

            OWidth, OHeight  = get_img_shape(row['filepath'])

            # logger.debug(image_raw.shape[:3])
            # logger.debug(image_id)
            # logger.debug(boxs)
            # logger.debug(row['image_path'])

            #### bbox conversion & normalization: range is [0,1]
            xmins=[]
            ymins=[]
            xmaxs=[]
            ymaxs=[]
            box_labels = []
            if boxs == boxs: ## check nan
                for box in ast.literal_eval(boxs):
                    xmins.append(box["x"] / OWidth)
                    ymins.append(box["y"] / OHeight)
                    xmaxs.append((box["width"] + box["x"]) / OWidth)
                    ymaxs.append((box["height"] + box["y"]) / OHeight)
                    
                    box_labels.append(label)
            else:
                continue                    
            # logger.debug(bbox)

            object_num = len(xmins)
            # logger.debug(object_num)

            feature  = train_serialize_example(
                image_raw.tostring(),
                str.encode(image_id),
                object_num,
                xmins,
                ymins,
                xmaxs,
                ymaxs,
                box_labels,
                label,
                )
            writer.write(feature)

        if show:
            filepath = 'train%.2i-%i.tfrecords'%(fold,fold_df.shape[0])
            filename = filepath.split('/')[-1]
            filesize = os.path.getsize(filepath)/10**6
            print(filename,':',np.around(filesize, 2),'MB')
'''