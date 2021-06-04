import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

from config import TRAINING_PATH, VALIDATION_PATH, TEST_PATH

train_files = os.listdir(TRAINING_PATH)
val_files = os.listdir(VALIDATION_PATH)
test_files = os.listdir(TEST_PATH)


def load_png(inputImageFileName):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("PNGImageIO")
    reader.SetFileName(inputImageFileName)
    image = reader.Execute()
    return image


def get_shapes():
    shapes = []
    for p in [TRAINING_PATH, VALIDATION_PATH, TEST_PATH]:
        for f in os.listdir(p):
            if f.count('.png') == 0:
                continue
            img = load_png(os.path.join(p, f))
            shapes.append(img.GetSize())

    return shapes

a = get_shapes()
a_arr = np.array(a)
plt.hist2d(a_arr[:, 0], a_arr[:, 1])
plt.show()