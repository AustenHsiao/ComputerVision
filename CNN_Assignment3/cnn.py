# This is just a really long script for the assignment. I did not make this modular because this was the easiest way for me to 
# learn the framework. I originally planned to refactor all of this, but eventually decided against it due to time...
# sorry to make you read this in its current form.

# written by Austen Hsiao for Assignment 3

from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
import tensorflow as tf
import cv2
from keras import layers
from keras import models
from tensorflow import keras
from keras.models import Model


def normalize(image):
    minn = image.min()
    maxx = image.max()
    return np.divide(np.subtract(image, minn), (maxx-minn))

# step 1: Load pretrained InceptionResNetV2. Report summary() and visualize the first layer filters.
pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
#pre_model.summary()

firstFilter = pre_model.get_layer(name='conv2d')
#store = firstFilter.get_weights()[0]
#f_min, f_max = store.min(), store.max()
# store = (store - f_min) / (f_max - f_min) #normalize
# store = store * 256 # scale up to 255
# for ilayer in range(32):
#    for rgb in range(3):
#        cv2.imwrite(f"firstlayer{rgb}_{ilayer}.png", store[:,:,rgb,ilayer])


# step 2: resize and preprocess (standardize)
catDogSet = tf.keras.preprocessing.image_dataset_from_directory(directory='dataset_3/dog vs cat/dataset/training_set', color_mode='rgb', labels="inferred", label_mode="binary", image_size=(150, 150), shuffle=True)  # load from training set
# Rather than scale to the highest value in each image, approximate this value using the max (255)
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
catDogSet = catDogSet.map(lambda x, y: (normalization_layer(x), y))

# step 3: Create transfer head
model = models.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
pre_model.trainable = False

# step 4: train and run/evaluate
#part i
testSet = tf.keras.preprocessing.image_dataset_from_directory(directory='dataset_3/dog vs cat/dataset/test_set', color_mode='rgb', labels="inferred", label_mode='binary', image_size=(150, 150))  # load from test set
testSet = testSet.map(lambda x, y: (normalization_layer(x), y))
#model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy())  # add
#result = model.evaluate(testSet)

predictions = np.array([])
labels = np.array([])
for sample, label in testSet.as_numpy_iterator():
    predict = model.predict(sample)
    predictions1 = []
    for predicti in predict.flatten():
        if predicti >= 0.5:
            predictions1.append(1)
        else:
            predictions1.append(0)
    predictions = np.concatenate([predictions, np.array(predictions1)])
    labels = np.concatenate([labels, label.flatten()])

total = 0
hit = 0
for p, a in zip(predictions, labels):
    if p == a:
        hit += 1
    total += 1
print(f'Accuracy: {hit*1.0/total}')

conf_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions, num_classes=2)
print(f'Confusion Matrix (Untrained): \n{conf_matrix}')

# part ii
model.compile(optimizer='SGD', loss=tf.keras.losses.binary_crossentropy)
history = model.fit(x=catDogSet, verbose=2, epochs=2)

predictions = np.array([])
labels = np.array([])
for sample, label in testSet.as_numpy_iterator():
    predict = model.predict(sample)
    predictions1 = []
    for predicti in predict.flatten():
        if predicti >= 0.5:
            predictions1.append(1)
        else:
            predictions1.append(0)
    predictions = np.concatenate([predictions, np.array(predictions1)])
    labels = np.concatenate([labels, label.flatten()])

total = 0
hit = 0
for p, a in zip(predictions, labels):
    if p == a:
        hit += 1
    total += 1
print(f'Accuracy: {hit*1.0/total}')

conf_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions, num_classes=2)
print(f'Confusion Matrix (Final Trained): \n{conf_matrix}')

# part iii #remake transfer head. use a subnetwork of original base.
# I scrolled around the summary() output to find a random pooling layer. A pool layer
# is chosen because I know it's scaling down after finishing some convolutions.
# The total number of layers is 780 and the pooling 2 layer occurs on layers 273,
# so we want to scale back the last layer: 
sub_model = Model(pre_model.input, pre_model.layers[-507].output)
#sub_model.summary()
sub_model.trainable = False
model1 = models.Sequential()
model1.add(sub_model)
model1.add(layers.Flatten())
model1.add(layers.Dense(256, activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))
#model1.summary()

model1.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy())

predictions = np.array([])
labels = np.array([])
for sample, label in testSet.as_numpy_iterator():
    predict = model1.predict(sample)
    predictions1 = []
    for predicti in predict.flatten():
        if predicti >= 0.5:
            predictions1.append(1)
        else:
            predictions1.append(0)
    predictions = np.concatenate([predictions, np.array(predictions1)])
    labels = np.concatenate([labels, label.flatten()])

total = 0
hit = 0
for p, a in zip(predictions, labels):
    if p == a:
        hit += 1
    total += 1
print(f'Accuracy: {hit*1.0/total}')
conf_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions, num_classes=2)
print(f'Confusion Matrix (Final Trained sub model): \n{conf_matrix}')
