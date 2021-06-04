import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
from skimage.io import imsave
import png
import numpngw

from config import TRAINING_PATH, VALIDATION_PATH, TEST_PATH, OUT_PATH

train_files = os.listdir(TRAINING_PATH)
val_files = os.listdir(VALIDATION_PATH)
test_files = os.listdir(TEST_PATH)

max_shape = (541, 865)
niklas_shape = (224, 224)


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


def save_png(image, outputImageFileName):
    sitk.WriteImage(image, outputImageFileName)
    # writer = sitk.ImageFileWriter()
    # writer.SetFileName(outputImageFileName)
    # writer.Execute(image)


def central_pad_zero(image):
    array = sitk.GetArrayFromImage(image)

    print(array.shape)
    x_ratio = array.shape[0] / niklas_shape[0]
    y_ratio = array.shape[1] / niklas_shape[1]
    print(x_ratio, y_ratio)
    #if array.shape[0] > niklas_shape[0] or array.shape[1] > niklas_shape[1]:
    # downsampling
    if x_ratio > y_ratio:
        y = int(np.round(array.shape[1] / x_ratio))
        x = niklas_shape[0]
    else:
        x = int(np.round(array.shape[0] / y_ratio))
        y = niklas_shape[1]
    print(x, y)

    img = Image.fromarray(array.astype(np.uint16))
    img = img.resize((y, x))
    array = np.array(img).astype(np.uint16)
    # array = np.resize(array, (x, y))

    full_array = np.zeros(niklas_shape)
    start_index_x = int((niklas_shape[0] - array.shape[0])/2)
    start_index_y = int((niklas_shape[1] - array.shape[1])/2)
    stop_index_x = start_index_x + array.shape[0]
    stop_index_y = start_index_y + array.shape[1]
    print(array.shape)
    print(start_index_x, stop_index_x, start_index_y, stop_index_y)
    full_array[start_index_x:stop_index_x, start_index_y:stop_index_y] = array
    #plt.imshow(full_array)
    #plt.show()
    print(full_array.shape)
    print(full_array.dtype)
    image = sitk.GetImageFromArray(full_array)
    print(image.GetSize())
    return full_array


def resize_all():
    # shapes = []
    for p in [TRAINING_PATH, VALIDATION_PATH, TEST_PATH]:
        for f in os.listdir(p):
            if f.count('.png') == 0:
                continue
            img = load_png(os.path.join(p, f))
            full_array = central_pad_zero(img)
            #full_array = np.stack((full_array, full_array, full_array), axis=2)
            print(full_array.shape)
            name = f"{f.split('.png')[0]}.png"
            outputImageFileName = os.path.join(p + '_out', name)
            img = Image.fromarray(full_array.astype(np.uint16))
            #imsave(outputImageFileName, full_array)
            #scipy.misc.imsave(outputImageFileName, full_array)
            # img.convert('RGB').save(outputImageFileName, 'PNG')
            #img.convert('RGB').save(outputImageFileName, 'JPEG', quality=95)
            #save_png(img, outputImageFileName)
            # Use pypng to write z as a color PNG.
            z = np.stack((full_array, full_array, full_array), axis=2)
            # z = np.concatenate((full_array, full_array, full_array), axis=1)
            print(z.shape)
            bit_depth = 16
            # png.from_array(img, mode='RGB').save('tmp.png')
            # png.from_array(z * 2 ** (bit_depth - 1), 'RGB;%s' % bit_depth).save(outputImageFileName)

            # Convert y to 16 bit unsigned integers.
            print(z.min())
            print(z.max())
            y = (65535 * ((z - z.min()) / (z.max() - z.min()))).astype(np.uint16)

            # Use numpngw to write z as a color PNG.
            numpngw.write_png(outputImageFileName, y)


'''
a = get_shapes()
a_arr = np.array(a)
plt.hist2d(a_arr[:, 0], a_arr[:, 1], bins=[200, 300],  range=[[0, 200], [0, 300]])
plt.show()
'''