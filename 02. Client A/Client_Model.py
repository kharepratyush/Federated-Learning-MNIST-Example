from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import utils
from tensorflow.keras import optimizers

import numpy as np
import glob
from os import path


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

def process_data():
    print("Processing data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_test.shape[0], 'test samples')

    y_train = utils.np_utils.to_categorical(y_train, num_classes)
    y_test = utils.np_utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

def build_model(x_train, y_train, x_test, y_test):

    if path.exists("agg_model.npy"):
        print("Agg weights exists...\nLoading weights...")
    else:
        print("No agg model found!\nBuilding weights...")
        
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])
                  
    if path.exists("agg_model.npy"):
        weights = np.load("agg_model.npy")
        #print(len(weights))
        for i in weights:
            pass
            #print(i.shape)
        model.set_weights(weights)

    np.load = np_load_old
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    return model

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def save_local_model_update(model):
    np.save("agg_model.npy", model.get_weights())
    print("Local model update written to local storage!")

def client_train():
    x_train, x_test, y_train, y_test =  process_data()
    model = build_model(x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    save_local_model_update(model)
