import tensorflow as tf
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Input
import matplotlib

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

    #########################
    # Train
    #########################
    CAE.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy)
    CAE.fit(x_train, x_train, batch_size=64, verbose=2, epochs=5, validation_data=(x_test, x_test))
    after_images = CAE.predict(x_test)
    
    #########################
    # Display before/after (4 samples, non-noisy) 
    #########################
    cv2.imwrite("img88.png", np.multiply(x_test[88], 255))
    cv2.imwrite("img88_CAE.png", np.multiply(after_images[88].reshape(28,28), 255))

    cv2.imwrite("img522.png", np.multiply(x_test[522], 255))
    cv2.imwrite("img522_CAE.png", np.multiply(after_images[522].reshape(28,28), 255))

    cv2.imwrite("img8888.png", np.multiply(x_test[8888], 255))
    cv2.imwrite("img8888_CAE.png", np.multiply(after_images[8888].reshape(28,28), 255))

    cv2.imwrite("img9001.png", np.multiply(x_test[9001], 255))
    cv2.imwrite("img9001_CAE.png", np.multiply(after_images[9001].reshape(28,28), 255))
    
    #########################
    #  Adding Gaussian noise to images
    #########################
    # x_test.reshape(10000,28,28,1)
    # x_train.reshape(60000,28,28,1)  
    x_train_noise = np.clip(np.array([np.add(img, np.random.normal(np.mean(img), np.std(img), (28,28,1))) for img in x_train]), 0.0, 1.0)
    x_test_noise  = np.clip(np.array([np.add(img, np.random.normal(np.mean(img), np.std(img), (28,28,1))) for img in x_test]), 0.0, 1.0)

    #########################
    #  Create and train using noisy images
    #########################
    input_img1 = Input(shape=(28,28,1))

    y = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(input_img1)
    y = MaxPooling2D(pool_size=2, padding='same')(y)
    y = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(y)
    encode1 = MaxPooling2D(pool_size=2, padding='same')(y)

    y = Flatten()(encode1)
    bottleNeck1 = Dense(2, activation='relu')(y)

    y = Dense(1568, activation='relu')(bottleNeck1)
    y = Reshape((7,7,32))(y)
    y = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(y)
    y = UpSampling2D()(y)
    y = Convolution2D(filters=32, kernel_size=3, activation='relu', padding='same')(y)
    y = UpSampling2D()(y)
    decoded1 = Convolution2D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(y)

    noiseCAE = Model(input_img1, decoded1)
    noiseCAE.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy)
    noiseCAE.fit(x_train_noise, x_train, batch_size=64, verbose=2, epochs=5, validation_data=(x_test_noise, x_test))
    after_images = noiseCAE.predict(x_test_noise)

    #########################
    # Display before/after (4 samples, noisy) 
    #########################
    cv2.imwrite("img88_noise.png", np.multiply(x_test_noise[88], 255))
    cv2.imwrite("img88_noise_CAE.png", np.multiply(after_images[88].reshape(28,28), 25), 255)

    cv2.imwrite("img522_noise.png", np.multiply(x_test_noise[522], 255))
    cv2.imwrite("img522_noise_CAE.png", np.multiply(after_images[522].reshape(28,28), 255))

    cv2.imwrite("img8888_noise.png", np.multiply(x_test_noise[8888], 255))
    cv2.imwrite("img8888_noise_CAE.png", np.multiply(after_images[8888].reshape(28,28), 255))

    cv2.imwrite("img9001_noise.png", np.multiply(x_test_noise[9001], 255))
    cv2.imwrite("img9001_noise_CAE.png", np.multiply(after_images[9001].reshape(28,28), 255))
