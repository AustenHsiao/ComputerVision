import tensorflow as tf
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Input

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

    x_train = x_train.reshape(60000,28,28,1)
    x_test  = x_test.reshape(10000,28,28,1)

    #########################
    # CAE Architecture
    #########################
    input_img = Input(shape=(28,28,1))

    x = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    encode = MaxPooling2D(pool_size=2, padding='same')(x)

    x = Flatten()(encode)
    bottleNeck = Dense(2, activation='relu')(x)

    x = Dense(1568, activation='relu')(bottleNeck)
    x = Reshape((7,7,32))(x)
    x = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    decoded = Convolution2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)

    CAE = Model(input_img, decoded)
    CAE.summary()

    #########################
    # Train
    #########################
    CAE.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy)
    CAE.fit(x_train, x_train, batch_size=64, verbose=2, epochs=5, validation_data=(x_test, x_test))
    after_images = CAE.predict(x_test)
    
    #########################
    # Display before/after (4 samples) 
    #########################
    cv2.imshow("#88", x_test[88])
    cv2.imshow("#88.2", after_images[88].reshape(28,28))

    cv2.imshow("#522", x_test[522])
    cv2.imshow("#522.2", after_images[522].reshape(28,28))

    cv2.imshow("#8888", x_test[8888])
    cv2.imshow("#8888.2", after_images[8888].reshape(28,28))

    cv2.imshow("#9001", x_test[9001])
    cv2.imshow("#9001.2", after_images[9001].reshape(28,28))
    cv2.waitKey(0)
    print("hi")