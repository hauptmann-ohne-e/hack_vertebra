import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def confusion(nc, pred, truth=None, scores=False):
    if not truth is None and not len(pred == len(truth)):
        return
    l = len(pred)

    # TD simple test should give 1.25
    # for i in range(l):
    #    pred[i] = 1.5

    gs = np.zeros((nc + 1, nc + 1))
    sc = np.zeros((nc + 1, nc + 1))
    gs[nc, nc] = len(pred)
    # Prediciton
    for i in range(l):
        gs[nc, max(min(round(pred[i]), nc - 1), 0)] += 1
    if truth is None:
        return gs[nc]
    # Truth
    for i in range(l):
        gs[max(min(round(truth[i]), nc - 1), 0), nc] += 1
    # Confusion
    for i in range(l):
        gs[max(min(round(truth[i]), nc - 1), 0), max(min(round(pred[i]), nc - 1), 0)] += 1
        if scores:
            sc[max(min(round(truth[i]), nc - 1), 0), max(min(round(pred[i]), nc - 1), 0)] += (truth[i] - pred[i]) * (
                    truth[i] - pred[i])

    # TD Calculate challenge score as defined by EY
    if scores:
        for i in range(nc):
            for j in range(nc):
                sc[i, nc] += sc[i, j]
        # print(sc)
        for i in range(nc):
            if sc[i, nc] > 0:  # Also means gs[i,nc] is zero!
                sc[i, nc] /= gs[i, nc]
                sc[nc, nc] += sc[i, nc]
        # print(sc)
        sc[nc, nc] /= nc
        # print(sc)
        print("EY(GS) Score: {:.2f}".format(sc[nc, nc]))

    cols = [_ for _ in range(nc)]
    cols.append("T")
    rows = [_ for _ in range(nc)]
    rows.append("P")
    gs_df = pd.DataFrame(gs, index=rows, columns=cols)
    return gs_df.astype("int64")


def label_regression_scale(labels):
    labels.replace({0.0: 0.0, 1.0: 20.0, 2.0: 30.0, 3.0: 50.0}, inplace=True)
    return labels


def pred_regression_scaleback(pred):
    for i in range(len(pred)):
        if pred[i] < 2:
            pred[i] = pred[i] / 2.0
        else:
            pred[i] = (pred[i] - 10.0) / 10.0  # might be casted to int()
    return pred
