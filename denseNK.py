import os
import json
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Activation, Dense, Flatten
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adagrad, Nadam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, BatchNormalization, MaxPooling2D


def create_denseNet(use_imagenet: bool = True,
                    input_shape: tuple = (224, 224, 3),
                    trainable: bool = False,
                    num_neurons: int = 256,
                    dropout: float = 0.3):
    """

    :param use_imagenet:
    :param input_shape:
    :param trainable:
    :param num_neurons:
    :param dropout:
    :return:
    """

    weights = None

    if use_imagenet:
        weights = "imagenet"

    # load the Densenet121 without the Classification Layers, but with the pre-trained weights from the imagenet
    # dataset
    densenet_without_fc = DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )

    # -------------------------------------- Relative Good Model ---------------------------------------------------

    if not trainable:
        # freeze the layer in Densnet121
        for idx, layer in enumerate(densenet_without_fc.layers):
            if idx < 137: #53
                layer.trainable = False

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in densenet_without_fc.layers])

    x = layer_dict['pool3_pool'].output
    x = Flatten()(x)
    x = Dense(units=num_neurons, name='FC-Layer', activation='relu')(x)
    if dropout > 0.0:
        x = Dropout(dropout)(x)
    preds = Dense(4, activation="softmax")(x)

    custom_model = Model(inputs=densenet_without_fc.input, outputs=preds)
    return custom_model
