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

from meanAveragePrecision import computeMeanAveragePrecision
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from keras.losses import MeanAbsoluteError,MeanSquaredError

train_bsz = 32
epochs = 1
lr = 0.0005
regression = False

def pr_function(image):
    img = np.array(image)
    print(np.max(img))
    return img


from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def show_examples(generator, r, c):
    x, y = generator.next()
    image = x
    label = y
    print(image.shape)
    print(image[0])
    plt.figure(figsize=(20, 20))
    for i in range(0, (r * c)):
        plt.subplot(r, c, i + 1)
        plt.title(label[i])
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

    y_col="grade"
    if (not regression):
        y_col=["grade_0.0", "grade_1.0", "grade_2.0", "grade_3.0"]

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(dataset_train['grade'].values),
                                                      dataset_train['grade'].values.astype(int))
    class_weights = dict(enumerate(np.array(class_weights)))
    #class_weights[1]*= 10.0;
    #class_weights[2]*= 10.0;
    #class_weights[3]*= 10.0;
  
    dataset_validation = pd.read_csv(CSV_VALIDATION_PATH,
                                     comment='\t', sep=',', skipinitialspace=True, header=0).dropna()
    dataset_test = pd.read_csv(CSV_TEST_PATH, na_values='?',
                               comment='\t', sep=',', skipinitialspace=True, header=0)

    if (not regression):
        dataset_train = pd.get_dummies(dataset_train, columns=['grade'])
        dataset_validation = pd.get_dummies(dataset_validation, columns=['grade'])

    train_datagen = ImageDataGenerator(  # preprocessing_function=pr_function,
        rescale=1. / 255.,
        rotation_range=10,
        zoom_range=0.1,
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
        y_col=y_col,
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
        y_col=y_col,
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
    net = create_denseNet(regression=regression)

    # Compile Model
    optimizer = Adam(learning_rate=0.0005)
    
    loss_regression = MeanAbsoluteError() # MeanSquaredError
    loss_categorical = "categorical_crossentropy"
        
    metrics_categorical = ["accuracy",
               tf.keras.metrics.AUC(curve="PR", name="APS", multi_label=True),
               tf.keras.metrics.AUC(curve="ROC", name="ROC-AUC", multi_label=True),
               #tf.keras.metrics.CategoricalAccuracy(),
               #tf.keras.metrics.TruePositives(),
               #tf.keras.metrics.TrueNegatives(),
               #tf.keras.metrics.FalsePositives(),
               #tf.keras.metrics.FalseNegatives()
               ]
                
    metrics_regression = [MeanAbsoluteError(),
                          MeanSquaredError(),
                         ]
    
    if (regression):
        net.compile(optimizer=optimizer,
                    loss=loss_regression,
                    metrics=metrics_regression,
                    )
#        net.summary()
        record = net.fit(train_generator,
                     validation_data=val_generator,
                     epochs=epochs,
                     verbose=1
                     ).history
    else:
        net.compile(optimizer=optimizer,
                    loss=loss_categorical,
                    metrics=metrics_categorical,
                    weighted_metrics=["accuracy"]
                    )

#        net.summary()
        record = net.fit(train_generator,
                     validation_data=val_generator,
                     epochs=epochs,
                     verbose=1,
                     class_weight=class_weights,
                     shuffle=True,
                     # callbacks=callbacks
                     ).history

    plotting_history_1(record, os.path.join(OUT_PATH,"training.png"),
                       f=customize_axis_plotting("loss"))

    val_generator.shuffle = False

    if (not regression):
        test_Y = np.array([np.where(r == 1)[0][0] for r in val_generator.labels])

        pred = net.predict(val_generator, verbose=1)
        print(multilabel_confusion_matrix(test_Y, np.argmax(pred, -1)))

        #print(np.average(test_Y-np.argmax(pred,-1)))
    
        t = pd.DataFrame(data=[test_Y,np.argmax(pred,-1),test_Y-np.argmax(pred,-1)]).transpose()
        t.to_csv("t.csv")    

        report = classification_report(test_Y, np.argmax(pred, -1), output_dict=True)
        df = pd.DataFrame(report).transpose()
        p = computeMeanAveragePrecision(test_Y, net.predict(val_generator))
        df["Mean_average_percision"] = np.concatenate([p[1], np.array([-1, -1, p[0]])])
        df.to_csv("results_on_val.csv")

    # Prediction
    val_ids = list(test_generator.filenames)
    #print(val_ids)

    pred = net.predict(test_generator, verbose=1)
    
    if (not regression):
        df = pd.DataFrame(list(zip(val_ids, np.argmax(pred, -1))), columns=["ID", "Prediction"])
    else:
        df = pd.DataFrame(list(zip(val_ids, pred)), columns=["ID", "Prediction"])
    print(df.head())
    df.to_csv('results.csv',
              index=False,
              header=False)


if __name__ == '__main__':
    main()
    pass
