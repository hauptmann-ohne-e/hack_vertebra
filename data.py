import tensorflow as tf
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

from config import CSV_TRAINING_PATH, CSV_VALIDATION_PATH, CSV_TEST_PATH, TEST_PATH, VALIDATION_PATH, TRAINING_PATH

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


# tf.keras.preprocessing.image_dataset_from_directory(
#     directory,
#     labels="inferred",
#     label_mode="int",
#     class_names=None,
#     color_mode="rgb",
#     batch_size=32,
#     image_size=(256, 256),
#     shuffle=True,
#     seed=None,
#     validation_split=None,
#     subset=None,
#     interpolation="bilinear",
#     follow_links=False,
#     smart_resize=False,
# )

import matplotlib.pyplot as plt


def show_examples(generator, r, c):
    x, y = generator.next()
    image = x
    print(image.shape)
    plt.figure(figsize=(20, 20))
    for i in range(0, (r * c)):
        plt.subplot(r, c, i + 1)
    plt.imshow(image[i])
    plt.show()


def main():
    # Make numpy printouts easier to read.
    np.set_printoptions(precision=3, suppress=True)

    # https://archive.ics.uci.edu/ml/datasets/Auto+MPG
    # column_names = ['img', 'grade']

    dataset_train = pd.read_csv(CSV_TRAINING_PATH, na_values='?',
                                comment='\t', sep=',', skipinitialspace=True, header=0)
    dataset_validation = pd.read_csv(CSV_VALIDATION_PATH, na_values='?',
                                     comment='\t', sep=',', skipinitialspace=True, header=0)
    dataset_test = pd.read_csv(CSV_TEST_PATH, na_values='?',
                              comment='\t', sep=',', skipinitialspace=True, header=0)

    dataset_train.tail()

    # print(dataset_train)

    datagen = ImageDataGenerator(rescale=1. / 65536.)
    test_datagen = ImageDataGenerator(rescale=1. / 65536.)

    train_generator = datagen.flow_from_dataframe(
        dataframe=dataset_train,
        directory=TRAINING_PATH,
        validate_filenames=False,
        x_col="img",
        y_col="grade",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(32, 32))

    valid_generator = datagen.flow_from_dataframe(
        dataframe=dataset_validation,
        directory=VALIDATION_PATH,
        validate_filenames=False,
        x_col="img",
        y_col="grade",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="raw",
        target_size=(32, 32))

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=dataset_test,
        directory=TEST_PATH,
        validate_filenames=False,
        x_col="img",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode=None,
        target_size=(32, 32))

    # test_generator = test_datagen.flow_from_directory(
    #     directory=TEST_PATH,
    #     batch_size=32,
    #     seed=42,
    #     shuffle=False,
    #     class_mode=None,
    #     target_size=(32, 32))

    # todo: clean data???
    # dataset = dataset_train.dropna()

    show_examples(train_generator, 3, 3)


if __name__ == '__main__':
    main()
    pass
