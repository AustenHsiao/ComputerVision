from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.utils import to_categorical
import cv2
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    ## Make model and train it on the Mnist digits
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        x_train = np.divide(x_train, 255.0).reshape(60000, 28, 28, 1)
        y_train = to_categorical(y_train)
        x_test = np.divide(x_test, 255.0).reshape(10000, 28, 28, 1)
        y_test = to_categorical(y_test)

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(100, activation='sigmoid'))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(10, activation='softmax'))
        model.summary()

        model.compile(optimizer='SGD', loss=tf.keras.losses.categorical_crossentropy)
        model.fit(x_train, y_train, batch_size=64, verbose=2, epochs=100)
    
        # Create confusion matrix
