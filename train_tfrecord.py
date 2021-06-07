## logging setup
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

import os
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





### parameters ###
# tf.config.list_physical_devices('GPU')
batch_size = 16
epochs = 30
num_classes = 4
# 複数のTFRecordを使う場合、以下の件数は全ファイルの合計になることに注意。
num_records_train = 1267*4
num_records_test  = 1266
# 1エポックあたりのミニバッチ数。学習時に使う。
steps_per_epoch_train = (num_records_train-1) // batch_size + 1
steps_per_epoch_test  = (num_records_test-1) // batch_size + 1

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
    image = tf.image.resize(image, [256,256])
    image = image / 255.0
    label = tf.cast(parsed["image/label"], tf.int32)

    xmin = tf.expand_dims(parsed["image/object/bbox/xmin"].values, 0)
    ymin = tf.expand_dims(parsed["image/object/bbox/ymin"].values, 0)
    xmax = tf.expand_dims(parsed["image/object/bbox/xmax"].values, 0)
    ymax = tf.expand_dims(parsed["image/object/bbox/ymax"].values, 0)
    # num_bboxes = tf.cast(parsed['image/object/count'], tf.int32)
    bboxes = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax])
    bboxes = tf.transpose(bboxes) # 4*N -> N*4
    bboxes = bboxes[0]

    return image, (label, bboxes)


dataset_train = tf.data.TFRecordDataset(["train{:02d}-1267.tfrecords".format(i) for i in range(4)]) \
    .map(parse_example) \
    .shuffle(num_records_train) \
    .batch(batch_size).repeat(-1)

dataset_val = tf.data.TFRecordDataset("train04-1266.tfrecords") \
    .map(parse_example) \
    .batch(batch_size)
# logger.debug(dataset_train)


### Simple model for multi-label object detection (detect single box only)###
layer_input = tf.keras.layers.Input(shape=(256,256,3),name="inputs")
flatten = tf.keras.layers.Flatten()(layer_input)
fc1 = tf.keras.layers.Dense(1024, activation="relu")(flatten)
fc2 = tf.keras.layers.Dense(512, activation="relu")(fc1)
# classification branch
fc_class = tf.keras.layers.Dense(256, activation="relu")(fc2)
class_output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_label")(fc_class)
# regression branch
fc_bbox = tf.keras.layers.Dense(256, activation="relu")(fc2)
bbox_output = tf.keras.layers.Dense(4, activation="sigmoid", name="bounding_box")(fc_bbox)

model = tf.keras.models.Model(
    inputs=layer_input,
    outputs=(class_output, bbox_output))


### Loss Definition ###
losses = {
	"class_label": "sparse_categorical_crossentropy",
	"bounding_box": "huber",
}
lossWeights = {
	"class_label": 1.0,
	"bounding_box": 1.0
}

model.compile(optimizer="adam",
              loss=losses,
              loss_weights=lossWeights,
              metrics=["accuracy"]
              )
model.summary()


### Trainig ###
# filepath="./retinanet/weights.{epoch:02d}-{loss:.4f}-{accuracy:.4f}.hdf5",
cp_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath="./retinanet/weights.{epoch:02d}-{loss:.4f}.hdf5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="auto")

learning_rates = np.linspace(0.01, 0.001, epochs)    
lr_cb = tf.keras.callbacks.LearningRateScheduler( \
    lambda epoch: float(learning_rates[epoch]))


H=model.fit(
    dataset_train,
    epochs=epochs,
    verbose=1,
    steps_per_epoch=steps_per_epoch_train,
    validation_data=dataset_val,
    callbacks=[cp_cb, lr_cb])
# validation_steps=1,

#### plot the total loss, label loss, and bounding box loss ####
import matplotlib.pyplot  as plt
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, epochs)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(N, H.history[l], label=l)
	ax[i].plot(N, H.history["val_" + l], label="val_" + l)
	ax[i].legend()
# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join(["losses.png"])
plt.savefig(plotPath)
plt.close()

# create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
	label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
	label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
# save the accuracies plot
plotPath = os.path.sep.join(["accs.png"])
plt.savefig(plotPath)