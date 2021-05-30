from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf

COMPRESSION_DEGREE = 2 # change this to change the compression factor (eg. 2 = k -> rank/2)

if __name__ == '__main__':
    ## Make model and train it on the Mnist digits
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
        x_train = np.divide(x_train, 255.0).reshape(60000, 28, 28, 1)
        y_train = to_categorical(y_train)
        x_test = np.divide(x_test, 255.0).reshape(10000, 28, 28, 1)
        y_test = to_categorical(y_test)

        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1), data_format='channels_last'))
        model.add(Dense(100, activation='sigmoid', input_shape=(784, 1)))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(10, activation='softmax'))
        model.summary()

        model.compile(optimizer='SGD', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=64, verbose=2, epochs=100, validation_data=(x_test, y_test))

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
        w_layer1 = layer1[0]
        w_layer2 = layer2[0]
        w_layer3 = layer3[0]
        b_layer1 = layer1[1]
        b_layer2 = layer2[1]
        b_layer3 = layer3[1]

        layer1_U, layer1_E, layer1_VT = np.linalg.svd(w_layer1, full_matrices=False)
        layer2_U, layer2_E, layer2_VT = np.linalg.svd(w_layer2, full_matrices=False)
        layer3_U, layer3_E, layer3_VT = np.linalg.svd(w_layer3, full_matrices=False)
        layer1_E = np.diag(layer1_E)
        layer2_E = np.diag(layer2_E)
        layer3_E = np.diag(layer3_E)

        layer1_k = int(layer1_U.shape[1] / COMPRESSION_DEGREE) 
        layer2_k = int(layer2_U.shape[1] / COMPRESSION_DEGREE)
        layer3_k = int(layer3_U.shape[1] / COMPRESSION_DEGREE)

        layer1_Uprime = np.matmul(layer1_U[:, :layer1_k], layer1_E[:layer1_k, :layer1_k])
        layer2_Uprime = np.matmul(layer2_U[:, :layer2_k], layer2_E[:layer2_k, :layer2_k])
        layer3_Uprime = np.matmul(layer3_U[:, :layer3_k], layer3_E[:layer3_k, :layer3_k])
        layer1_VTprime = layer1_VT[:layer1_k, :]
        layer2_VTprime = layer2_VT[:layer2_k, :]
        layer3_VTprime = layer3_VT[:layer3_k, :]

        # Create a new model where every dense layer in the original model is replaced by 2 dense layers
        # corresponding to the factors of SVD (above)
        modelCompressed = Sequential()
        modelCompressed.add(Flatten(input_shape=(28, 28, 1), data_format='channels_last'))
        modelCompressed.add(Dense(layer1_Uprime.shape[1], activation='sigmoid', input_shape=(784, 1)))
        modelCompressed.add(Dense(layer1_VTprime.shape[1], activation='sigmoid'))
        modelCompressed.add(Dense(layer2_Uprime.shape[1], activation='sigmoid'))
        modelCompressed.add(Dense(layer2_VTprime.shape[1], activation='sigmoid'))
        modelCompressed.add(Dense(layer3_Uprime.shape[1], activation='sigmoid'))
        modelCompressed.add(Dense(layer3_VTprime.shape[1], activation='softmax'))
        modelCompressed.summary()

        # copy over weights and bias weights from trained model SVD
        layersCompressed = modelCompressed.layers
        layersCompressed[1].set_weights([layer1_Uprime, b_layer1[:layer1_Uprime.shape[1]]])
        layersCompressed[2].set_weights([layer1_VTprime, b_layer1[:layer1_VTprime.shape[1]]])
        layersCompressed[3].set_weights([layer2_Uprime, b_layer2[:layer2_Uprime.shape[1]]])
        layersCompressed[4].set_weights([layer2_VTprime, b_layer2[:layer2_VTprime.shape[1]]])
        layersCompressed[5].set_weights([layer3_Uprime, b_layer3[:layer3_Uprime.shape[1]]])
        layersCompressed[6].set_weights([layer3_VTprime, b_layer3[:layer3_VTprime.shape[1]]])

    ## Compile and train compressed model
        modelCompressed.compile(optimizer='SGD', loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
        modelCompressed.fit(x_train, y_train, batch_size=64, verbose=2, epochs=10, validation_data=(x_test, y_test))

        # Create confusion matrix
        testPredictions_C = modelCompressed.predict(x_test)
        predictions_C = [np.argmax(test) for test in testPredictions_C]
        labels_C = [np.argmax(truth) for truth in y_test]
        conf_mat_C = tf.math.confusion_matrix(labels=labels_C, predictions=predictions_C, num_classes=10)
        print(conf_mat_C)
