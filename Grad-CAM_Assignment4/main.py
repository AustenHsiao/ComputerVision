import tensorflow as tf
import numpy as np
#import cv2
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, UpSampling2D, Reshape
#from keras.utils import np_utils

if __name__ == '__main__':
    #########################
    # Load Data
    #########################
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    #########################
    # Normalize
    #########################
    x_train = np.divide(x_train, 255.0)
    x_test = np.divide(x_test, 255.0)
    y_train = np.divide(y_train, 255.0)
    y_test = np.divide(y_test, 255.0)

    x_train = x_train.reshape(60000,28,28,1)
    x_test  = x_test.reshape(10000,28,28,1)

    
    #########################
    # CAE Architecture
    #########################
    encoder = Sequential(name='encoder')
    encoder.add(Convolution2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same'))
    encoder.add(MaxPooling2D(pool_size=2))
    encoder.add(Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    encoder.add(MaxPooling2D(pool_size=2))
    encoder.add(Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    #encoder.summary()

    middle = Sequential(name='bottleNeck')
    middle.add(Flatten(data_format='channels_first'))
    middle.add(Dense(2, activation='relu'))
    #middle.summary()

    decoder = Sequential(name='decoder')
    decoder.add(Dense(1568, activation='relu'))
    decoder.add(Reshape((32, 7, 7)))
    decoder.add(Convolution2D(filters=32, kernel_size=3, activation='relu', input_shape=(7, 7, 1), padding='same'))
    decoder.add(UpSampling2D())
    decoder.add(Convolution2D(filters=32, kernel_size=3, activation='relu', input_shape=(14, 14, 1), padding='same'))
    decoder.add(UpSampling2D())
    decoder.add(Convolution2D(filters=32, kernel_size=3, activation='relu', input_shape=(28, 28, 1), padding='same'))
    decoder.add() ##<----- ADD HERE
    #decoder.summary()

    CAE = Sequential()
    CAE.add(encoder)
    CAE.add(middle)
    CAE.add(decoder)
    CAE.summary()

    #########################
    # Train
    #########################
    CAE.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy)
    CAE.fit(x_train, y_train, batch_size=64, verbose=2, epochs=5)
    #print(x_train.shape)
