import matplotlib.pyplot as plt
import itertools

import pandas as pd
from tensorflow.keras import Model
from scipy import interpolate as interp
from sklearn.metrics import precision_recall_curve, f1_score, auc, roc_curve
from sklearn.metrics import classification_report, confusion_matrix
from plot_keras_history import plot_history

import numpy as np


def plot_average_ROC_Fold_Curve(y_true_list, y_prob_list, fold_size, pathname):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(fold_size):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_true=y_true_list[i], y_score=y_prob_list[i])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(pathname)
    plt.close()
    plt.clf()


def plot_pr_curve(y_proba, y_true, y_pred, pathname):
    """
    :param y_proba:
    :param y_true:
    :param y_pred:
    :param pathname:
    :return:
    """

    # calculate the precision-recall-curve
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_proba)
    lr_f1 = f1_score(y_true, y_pred)
    lr_auc = auc(lr_recall, lr_precision)

    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

    # plot the precision-recall curves
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(pathname)
    plt.close()
    plt.clf()

    return {'Precision': lr_precision, 'Recall': lr_recall, 'F1 Score:': lr_f1, "APS": lr_auc}


def plot_roc_curve(fpr_test, tpr_test, auc_test, pathname):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_test, tpr_test, label='ROC_Test (area = {:.3f})'.format(auc_test))
    # plt.plot(fpr_keras_val, tpr_keras_val, label='ROC_Val (area = {:.3f})'.format(auc_keras_val))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(pathname)
    plt.close()
    plt.clf()


def plot_confusion_matrix_1(cm, classes,
                            pathname,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(pathname)
    plt.close()
    plt.clf()


def plot_confusion_matrix(y_true, y_pred, classes,
                          pathname, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(pathname)
    plt.close()
    plt.clf()
    return ax


def plot_history_0(record: dict, pathname: str):
    """
    Plotting the keras history
    :param record: the history dictionary from keras fit method
    :param pathname: the name you want to save the ploting graphs
    :return: -
    """
    # Create count of the number of epochs
    epoch_count = range(1, len(record['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('History')

    # plot the training and validation loss
    ax1.set_title('Loss')
    ax1.plot(epoch_count, record['loss'], 'r--')
    ax1.plot(epoch_count, record['val_loss'], 'b-')
    ax1.legend(['Training Loss', 'Validation Loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # plot the training and validation accuracy
    ax2.set_title('Accuracy')
    ax2.plot(epoch_count, record['accuracy'], 'r--')
    ax2.plot(epoch_count, record['val_accuracy'], 'b-')
    ax2.legend(['Training Accuracy', 'Validation Accuracy'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    # fig.tight_layout()
    # save the figure
    fig.tight_layout()
    fig.savefig(pathname)
    plt.close(fig)
    plt.clf()


def customize_axis_plotting(mode: str) -> callable(object):
    """
    Create to customize the axis for the plotting
    :param mode:
    :return: a callable function to customize the axis for the plotting
    """
    if mode == "loss":
        def callback(axis):
            axis.legend(['Training', 'Validation'])

        return callback
    elif mode == "accuracy":
        def callback(axis):
            print("AXIS_loss")
            print(axis)

        return callback()
    return None


def plotting_history_1(history, filename, f):
    plot_history(history)
    plot_history(history, path=filename, customization_callback=f)
    plt.close()
    plt.clf()


def plot_imbalanced_dataset(true_label_list, pathname: str):
    import seaborn as sns

    unique, counts = np.unique(true_label_list, return_counts=True)
    temp_dict = dict(zip(unique, counts))

    dist = pd.Series(true_label_list)
    label_counts = dist.value_counts()
    plt.figure(figsize=(6, 4))
    sns.barplot(label_counts.index, label_counts.values, alpha=0.9)
    plt.xticks(rotation='vertical')
    plt.xlabel('Image Labels', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title('Imbalanced Data distribution')
    plt.savefig(pathname)
    plt.clf()
    plt.close()