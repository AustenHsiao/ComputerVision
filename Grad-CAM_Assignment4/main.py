import tensorflow as tf
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, UpSampling2D, Reshape
#from keras.utils import np_utils

if __name__ == '__main__':
    # Load Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize
    x_train = np.divide(x_train, 255.0)
    x_test = np.divide(x_test, 255.0)
    
    #cv2.imshow("train0", x_train[59999])
    #cv2.waitKey(0)
    
    # CAE Architecture
    encoder = Sequential(name='encoder')
    encoder.add(Convolution2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1, 28, 28), data_format='channels_first', padding='same'))
    encoder.add(MaxPooling2D(data_format='channels_first'))
    encoder.add(Convolution2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1, 14, 14), data_format='channels_first', padding='same'))
    encoder.add(MaxPooling2D(data_format='channels_first'))
    encoder.add(Convolution2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1, 7, 7), data_format='channels_first', padding='same'))
    #encoder.summary()

    middle = Sequential(name='bottleNeck')
    middle.add(Flatten(data_format='channels_first'))
    middle.add(Dense(2, activation='relu'))
    #middle.summary()

    decoder = Sequential(name='decoder')
    decoder.add(Dense(1568, activation='relu'))
    decoder.add(Reshape((32, 7, 7)))
    decoder.add(Convolution2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1, 7, 7), data_format='channels_first', padding='same'))
    decoder.add(UpSampling2D(data_format='channels_first'))
    decoder.add(Convolution2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1, 14, 14), data_format='channels_first', padding='same'))
    decoder.add(UpSampling2D(data_format='channels_first'))
    decoder.add(Convolution2D(32, (3, 3), strides=(1,1), activation='relu', input_shape=(1, 28, 28), data_format='channels_first', padding='same'))
    #decoder.summary()

    CAE = Sequential()
    CAE.add(encoder)
    CAE.add(middle)
    CAE.add(decoder)
    CAE.summary()
