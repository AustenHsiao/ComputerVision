from keras.applications.inception_resnet_v2 import InceptionResNetV2

pre_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
pre_model.summary()