import tensorflow as tf
from config import *

preprop = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_PATH,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode="grayscale")

for images, _ in preprop.take(100):
    print(images.shape)
