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
from tensorflow.python.ops import gen_sparse_ops
from denseNK import create_denseNet
from config import CSV_TRAINING_PATH, CSV_VALIDATION_PATH, CSV_TEST_PATH, TEST_PATH, VALIDATION_PATH, TRAINING_PATH, OUT_PATH
from plotting import plotting_history_1, customize_axis_plotting

from meanAveragePrecision import computeMeanAveragePrecision
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from keras.losses import MeanAbsoluteError,MeanSquaredError

from td_helper import show_examples,confusion,label_regression_scale,pred_regression_scaleback

train_bsz = 32
epochs = 5
lr = 0.0005

regression = False
rescale_regression = regression and False

def pr_function(image):
    img = np.array(image)
    print(np.max(img))
    return img

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

def main():
    # Make numpy printouts easier to read.
    np.set_printoptions(precision=3, suppress=True)

    # https://archive.ics.uci.edu/ml/datasets/Auto+MPG
    # column_names = ['img', 'grade']

    dataset_train = pd.read_csv(CSV_TRAINING_PATH,
                                comment='\t', sep=',', skipinitialspace=True, header=0).dropna()
    dataset_validation = pd.read_csv(CSV_VALIDATION_PATH,
                                     comment='\t', sep=',', skipinitialspace=True, header=0).dropna()
    dataset_test = pd.read_csv(CSV_TEST_PATH, na_values='?',
                               comment='\t', sep=',', skipinitialspace=True, header=0)

    orig_dataset_train = dataset_train.copy()
    orig_dataset_validation = dataset_validation.copy()
    orig_dataset_test = dataset_test.copy()

    print("Training Class Distribution")
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
    print("Class weights: {}".format(class_weights))
    
    if (rescale_regression):
        dataset_train["grade"] = label_regression_scale(dataset_train["grade"])
        dataset_validation["grade"] = label_regression_scale(dataset_validation["grade"])

    if (not regression): #Turn into one-hot coding
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

    #show_examples(train_generator, 4, 8)
    #show_examples(val_generator, 4, 8)

    # Create a custom model
    net = create_denseNet(regression=regression)

# Compile Model and Fitting

    optimizer = Adam(learning_rate=lr)
    
    loss_regression = MeanSquaredError() #MeanAbsoluteError()
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
        #net.summary()
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

        #net.summary()
        record = net.fit(train_generator,
                     validation_data=val_generator,
                     epochs=epochs,
                     verbose=1,
                     class_weight=class_weights,
                     shuffle=True,
                     ).history

    plotting_history_1(record, os.path.join(OUT_PATH,"training.png"),
                       f=customize_axis_plotting("loss"))

# Validation

    #TD This combination (!!!) will cause internal reset of index order without shuffling.
    val_generator.shuffle = False
    val_generator.index_array = None
    
    pred = net.predict(val_generator, verbose=1)
    if (not regression):
        #test_Y = np.array([np.where(r == 1)[0][0] for r in val_generator.labels])
        test_Y = orig_dataset_validation['grade'].values
        pred = np.argmax(pred, -1)
    else:
        test_Y = np.array([r for r in val_generator.labels])
        pred = np.ndarray.flatten(pred)
        if (rescale_regression):
            test_Y = orig_dataset_validation['grade'].values
            pred = pred_regression_scaleback(pred)
    pred = np.clip(pred,0,3)

    print(multilabel_confusion_matrix(test_Y, np.round(pred)))
    
    print(confusion(4,pred,test_Y,True))

    report = classification_report(test_Y, np.round(pred), output_dict=True)
    df = pd.DataFrame(report).transpose()
    if (not regression):
        p = computeMeanAveragePrecision(test_Y, net.predict(val_generator))
        df["Mean_average_percision"] = np.concatenate([p[1], np.array([-1, -1, p[0]])])
    df.to_csv("Report_on_Val.csv")

# Test Prediction

    val_ids = list(test_generator.filenames)
    pred = net.predict(test_generator, verbose=1)
    if (not regression):
        pred = np.argmax(pred, -1)
    else:
        pred = np.ndarray.flatten(pred)
        if (rescale_regression):
            pred = pred_regression_scaleback(pred)
    pred = np.clip(pred,0,3)

    print(confusion(4,pred))

# Write test predicitons to files [csv,json]
    
    df = pd.DataFrame(list(zip(val_ids, pred)), columns=["image", "prediction"])

    df.to_csv(os.path.join(OUT_PATH,'results.csv'),
              index=False,
              header=False)
           
    df.to_json(os.path.join(OUT_PATH,'results.json'),
               orient="records",
              )
    
if __name__ == '__main__':
    main()
    pass