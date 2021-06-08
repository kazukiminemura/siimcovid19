## logging setup
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes # get file type (built-in lib)
import argparse
import imutils
import pickle
import cv2
import ast
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input image/text file of image paths")
args = vars(ap.parse_args()) # return __dict__ attribute


filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]
if "text/plain" == filetype:
    imagePaths = open(args["input"].read().strip().split("\n"))

# load our object detector
logger.debug("[INFO] loading object detector...")
model = load_model("./retinanet/weights.30-0.8596.hdf5")

# get ground truth
import pandas as pd
train_df = pd.read_csv( \
    f'./input/siimcovid19-512-jpg-image-dataset/train.csv')
gts = train_df[["boxes", "image_id", "filepath"]]


# get dcm image info
import pydicom
def get_img_shape(path):
    """ Return (width, height) """
    dcm = pydicom.read_file(path)
    return dcm.Rows, dcm.Columns

for imagePath in imagePaths:
    image = load_img(imagePath, target_size=(256,256))
    # image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    (labelPreds, boxPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]
    
    label = np.argmax(labelPreds, axis=1)

    # load the inpur image
    image = cv2.imread(imagePath)
    image = imutils.resize(image,width=600)
    (h, w) = image.shape[:2]

    # scale the predicted boudnign box coordinates
    [startX, startY, endX, endY] = 
        np.array([startX, startY, endX, endY] * np.array([w, h, w, h]),
        dtype="int")

    # draw the predicted bounding box and class label
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, str(label), (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY),
        (0, 255, 255), 2)

    # draw ground truth boxes
    basename = os.path.basename(imagePath)[:-4]
    gt = gts[gts["image_id"] == basename]

    OWidth, OHeight  = get_img_shape(gt['filepath'].item())
    for box in ast.literal_eval(gt["boxes"].item()):
        xmins = int((box["x"] / OWidth) * w)
        ymins = int((box["y"] / OHeight) * h)
        xmaxs = int(((box["width"] + box["x"]) / OWidth) * w)
        ymaxs = int(((box["height"] + box["y"]) / OHeight) * h)
        cv2.rectangle(image, (xmins, ymins), (xmaxs, ymaxs),
            (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)

