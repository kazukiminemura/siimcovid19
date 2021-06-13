import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers


class EfficientNet(keras.Model):
    def __init__(self, num_classes, intput_image_size, **kwargs):
        super(EfficientNet, self).__init__(name="EfficientNet", **kwargs)
        self.num_classes = num_classes
        self.intput_image_size = intput_image_size



    def call(self, image, training=False):

        feature_extractor_url = \
            "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
        self.feature_extractor_layer = hub.KerasLayer(
            feature_extractor_url,
            input_shape=(self.intput_image_size,self.intput_image_size,3))
        # UnFreeze all layers
        for layer in feature_extractor_url.layers:
            layer.trainable = True


        freatues = self.feature_extractor_layer(image)
        # confidence output
        cls_outputs = layers.Dense( \
            num_classes, activation='softmax', name='class_label')(freatues)

        # model
        model = keras.Model(
            inputs = self.feature_extractor_layer.input,
            outputs = cls_outputs
        )
        
        return model