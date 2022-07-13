from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import ast
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np
import glob

num_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)


# In[10]:


def process_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test

def build_model(avg = None):
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

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer="adam",
              metrics=['accuracy'])

    if avg is not None:
        model.set_weights(avg)
        
    return model
    
def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def save_agg_model(model):
    #model.save_weights("agg_model.h5")
    np.save("agg_model.npy", model.get_weights())
    #print(len(model.get_weights()))
    #for i in model.get_weights():
    #    print(i.shape)
    print("Model written to storage!")

import h5py
def model_aggregation():
    _, x_test, _, y_test =  process_data()
    
    models = glob.glob("client_models_*.npy")
    print(models)
    
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    
    arr = []
    for i in models:
        print(i)
        weights = np.load(i)
        #print(len(weights))
        #for i in weights:
        #    print(i.shape)
        arr.append(weights)
        
    if len(arr) == 0:
        avg = None
    else:
        arr = np.array(arr)
        avg = np.average(arr, axis=0)
    
    #for i in avg:
    #    print(i.shape)
    np.load = np_load_old
    
    model = build_model(avg)
    evaluate_model(model, x_test, y_test)
    save_agg_model(model)