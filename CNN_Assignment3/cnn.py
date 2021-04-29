from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
import tensorflow as tf
import cv2
from keras import layers
from keras import models


def normalize(image):
    minn = image.min()
    maxx = image.max()
    return np.divide(np.subtract(image, minn), (maxx-minn))


# step 1: Load pretrained InceptionResNetV2. Report summary() and visualize the first layer filters.
pre_model = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3))
# pre_model.summary()
#firstFilter = pre_model.get_layer(name='conv2d')
#store = firstFilter.get_weights()[0]
#f_min, f_max = store.min(), store.max()
# store = (store - f_min) / (f_max - f_min) #normalize
# store = store * 256 # scale up to 255
# for ilayer in range(32):
#    for rgb in range(3):
#        cv2.imwrite(f"firstlayer{rgb}_{ilayer}.png", store[:,:,rgb,ilayer])


# step 2: resize and preprocess (standardize)
catDogSet = tf.keras.preprocessing.image_dataset_from_directory(
    directory='dataset_3/dog vs cat/dataset/training_set', color_mode='rgb', labels="inferred", label_mode="int", image_size=(150, 150), shuffle=True)  # load from training set
# Rather than scale to the highest value in each image, I approximate this value using the max (255)
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    1./255)
catDogSet = catDogSet.map(lambda x, y: (normalization_layer(x), y))

# for i in catDogSet.as_numpy_iterator():
# print(i[0])

# step 3:
model = models.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

pre_model.trainable = False
