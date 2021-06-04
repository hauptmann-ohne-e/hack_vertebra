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

def create_denseNet(model_dict):
    weights = None
    
    if model_dict['use_imagenet']:
        weights = model_dict['weights']

    # load the Densenet121 without the Classification Layers, but with the pre-trained weights from the imagenet
    # dataset
    densenet_without_fc = DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=model_dict['input_shape'],
    )

    densenet_without_fc.summary()

    # -------------------------------------- Relative Good Model ---------------------------------------------------

    if not model_dict['trainable']:
        # freeze the layer in Densnet121
        for idx, layer in enumerate(densenet_without_fc.layers):
            if idx < 137:
                layer.trainable = False

    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in densenet_without_fc.layers])

    x = layer_dict['pool3_pool'].output 
    x = Flatten()(x)
    x = Dense(model_dict['num_neurons'], name='FC-Layer', activation='relu')(x)
    if model_dict['dropout'] > 0:
        print("You use dropout: {}".format(model_dict['dropout']))
        x = Dropout(model_dict['dropout'])(x)
    #preds = Dense(1, activation='sigmoid')(x) #TD For regression, we MUST NOT use activation here
    preds = Dense(1)(x)
    
    custom_model = Model(inputs=densenet_without_fc.input, outputs=preds)
    return custom_model