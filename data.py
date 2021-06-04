import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.applications.densenet import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import save_model
from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.utils import class_weight
from denseNK import create_denseNet
from config import CSV_TRAINING_PATH, CSV_VALIDATION_PATH, CSV_TEST_PATH, TEST_PATH, VALIDATION_PATH, TRAINING_PATH, OUT_PATH
from plotting import plotting_history_1, customize_axis_plotting

def pr_function(image):
    img = np.array(image)
    print(np.max(img))
    return img


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


def show_examples(generator, r, c):
    x, y = generator.next()
    image = x
    print(image.shape)
    print(image[0])
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

    dataset_train = pd.read_csv(CSV_TRAINING_PATH,
                                comment='\t', sep=',', skipinitialspace=True, header=0).dropna()

    print("Class Distribution")
    print(dataset_train.groupby('grade').count())

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(dataset_train['grade'].values),
                                                      dataset_train['grade'].values.astype(int))
    class_weights = dict(enumerate(np.array(class_weights)))

    dataset_train = pd.get_dummies(dataset_train, columns=['grade'])
    dataset_validation = pd.read_csv(CSV_VALIDATION_PATH,
                                     comment='\t', sep=',', skipinitialspace=True, header=0).dropna()
    dataset_validation = pd.get_dummies(dataset_validation, columns=['grade'])
    dataset_test = pd.read_csv(CSV_TEST_PATH, na_values='?',
                               comment='\t', sep=',', skipinitialspace=True, header=0)

    train_bsz = 32
    epochs = 20
    lr = 0.0005

    train_datagen = ImageDataGenerator(  # preprocessing_function=pr_function,
        rescale=1. / 255.,
        rotation_range=20,
        #zoom_range=[1.3,1.7],
        brightness_range=[0.8,1.2],
    )
    val_test_datagen = ImageDataGenerator(  # preprocessing_function=pr_function,
        rescale=1. / 255.
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=dataset_train,
        directory=TRAINING_PATH,
        validate_filenames=False,
        x_col="img",
        y_col=["grade_0.0", "grade_1.0", "grade_2.0", "grade_3.0"],
        batch_size=train_bsz,
        seed=42,
        shuffle=True,
        class_mode="raw",
        color_mode="rgb",
        target_size=(224, 224))

    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=dataset_validation,
        directory=VALIDATION_PATH,
        validate_filenames=False,
        x_col="img",
        y_col=["grade_0.0", "grade_1.0", "grade_2.0", "grade_3.0"],
        batch_size=train_bsz, #1
        seed=42,
        shuffle=True,
        class_mode="raw",
        color_mode="rgb",
        target_size=(224, 224))

    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=dataset_test,
        directory=TEST_PATH,
        validate_filenames=False,
        x_col="img",
        batch_size=1,
        seed=42,
        shuffle=False,
        class_mode=None,
        color_mode="rgb",
        target_size=(224, 224))

    show_examples(train_generator, 4, 8)
    show_examples(val_generator, 4, 8)

    print("Class weights: {}".format(class_weights))

    # Create a custom model
    net = create_denseNet()

    # Compile Model
    optimizer = Adam(learning_rate=0.0005)
    loss = "categorical_crossentropy"
    metrics = ["accuracy",
               tf.keras.metrics.AUC(curve="PR", name="APS", multi_label=True),
               tf.keras.metrics.AUC(curve="ROC", name="ROC-AUC", multi_label=True),
               tf.keras.metrics.CategoricalAccuracy(),
               tf.keras.metrics.TruePositives(),
               tf.keras.metrics.TrueNegatives(),
               tf.keras.metrics.FalsePositives(),
               tf.keras.metrics.FalseNegatives()
               ]

    net.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics,
                weighted_metrics=["accuracy"]
                )

    net.summary()

    record = net.fit(train_generator,
                     steps_per_epoch=train_generator.samples // train_bsz,
                     validation_data=val_generator,
                     validation_steps=val_generator.samples // val_generator.batch_size,
                     epochs=epochs,
                     verbose=1,
                     class_weight=class_weights,
                     shuffle=True,
                     # callbacks=callbacks
                     ).history

    plotting_history_1(record, os.path.join(OUT_PATH,"training.png"),
                       f=customize_axis_plotting("loss"))

    # Prediction
    val_ids = list(test_generator.filenames)
    #print(val_ids)

    pred = net.predict(test_generator, verbose=1)
    df = pd.DataFrame(list(zip(val_ids, np.argmax(pred, -1))), columns=["ID", "Prediction"])
    print(df.head())
    df.to_csv('result_for_fold_val.csv',
              index=False,
              header=False)


if __name__ == '__main__':
    main()
    pass
