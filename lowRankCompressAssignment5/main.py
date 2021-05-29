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
        model.add(Flatten(input_shape=(28, 28, 1), data_format='channels_last'))
        model.add(Dense(100, activation='sigmoid', input_shape=(784,1)))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(10, activation='softmax'))
        model.summary()

        model.compile(optimizer='SGD', loss=tf.keras.losses.categorical_crossentropy)
        model.fit(x_train, y_train, batch_size=64, verbose=2, epochs=100)
    
        # Create confusion matrix
        testPredictions = model.predict(x_test)
        predictions = [np.argmax(test) for test in testPredictions]
        labels = [np.argmax(truth) for truth in y_test]
        conf_mat = tf.math.confusion_matrix(labels=labels, predictions=predictions, num_classes=10)
        print(conf_mat)

    ## Generate the Low-Rank Model
        layers = model.layers
        layer1 = layers[1].get_weights()
        layer2 = layers[2].get_weights()
        layer3 = layers[3].get_weights()
        w_l1 = layer1[0]
        w_l2 = layer2[0]
        w_l3 = layer3[0]
        w_l1_b = layer1[1]
        w_l2_b = layer2[1]
        w_l3_b = layer3[1]