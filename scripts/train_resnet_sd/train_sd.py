# Training procedure for CIFAR-10/CIFAR-100 using ResNet 110 (SD)

import numpy as np
import collections
import pickle

import keras
from train_resnet_sd.resnet_sd import resnet_sd_model
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10, cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model

# constants
EPOCHS = 500
BATCH_SIZE = 128
PATH_W = path.join("..", "..", "models")
PATH_L = path.join("..", "..", "logits")


# Learning rate schedule
def lr_sch(epoch):
    if epoch < EPOCHS * 0.5:
        return 0.1
    elif epoch < EPOCHS * 0.75:
        return 0.01
    else:
        return 0.001


# Per channel mean and std normalization
def color_preprocessing(x_train, x_val, x_test):
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')    
    x_test = x_test.astype('float32')
    
    mean = np.mean(x_train, axis=(0,1,2))  # Per channel mean
    std = np.std(x_train, axis=(0,1,2))
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    
    return x_train, x_val, x_test        
        
        
def prep_data(seed = 13, num_classes = 10):
    
    # load data
    if num_classes == 10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif num_classes == 100:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        print("Not supported number of classes: %i" % num_classes)
    
    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000, random_state=seed)  # random_state = seed

    
    # color preprocessing
    x_train45, x_val, y_train45, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)  # random_state = seed
    x_train45, x_val, x_test = color_preprocessing(x_train45, x_val, x_test)    
    
    y_train45 = keras.utils.to_categorical(y_train45, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    
    return ((x_train45, y_train45), (x_val, y_val), (x_test, y_test))


def train(seed = 13, num_classes = 10):


    (x_train45, y_train45), (x_val, y_val), (x_test, y_test) = prep_data(seed, num_classes)


    img_gen = ImageDataGenerator(
        horizontal_flip=True,
        data_format="channels_last",
        width_shift_range=0.125,  # 0.125*32 = 4 so max padding of 4 pixels, as described in paper.
        height_shift_range=0.125,
        fill_mode="constant",
        cval = 0
    )

    img_gen.fit(x_train45)
    
        
    # building and training net
    model = resnet_sd_model(img_shape = (32,32), img_channels = 3, 
                            layers = 110, nb_classes = num_classes, verbose = True)
    sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    checkpointer = ModelCheckpoint(path.join(PATH_W, 'weight_resnet110_SD_c%i_seed%i_best.hdf5' % (num_classes, seed)), verbose=1, save_best_only=True)


    hist = model.fit_generator(img_gen.flow(x_train45, y_train45, batch_size=BATCH_SIZE, shuffle=True),
                    steps_per_epoch=len(x_train45) // BATCH_SIZE,
                    validation_steps=len(x_val) // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data = (x_val, y_val),
                    callbacks=[LearningRateScheduler(lr_sch), checkpointer])

    model.save_weights(path.join(PATH_W, 'weights_resnet110_SD_c%i_seed%i.hdf5' % (num_classes, seed)))
    
    
    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_resnet110_SD_c%i_seed%i.p' % (num_classes, seed), 'wb') as f:
        pickle.dump(hist.history, f)
    
    
def gen_logits(seed = 13, num_classes = 13):

    weights_file = path.join(PATH_W, 'weights_resnet110_SD_c%i_seed%i.hdf5' % (num_classes, seed))  # TODO check whether weight file exists   

    (x_train45, y_train45), (x_val, y_val), (x_test, y_test) = prep_data(seed, num_classes)
    
    model = resnet_sd_model(img_shape = (32,32), img_channels = 3, 
                            layers = 110, nb_classes = num_classes, verbose = True)


    file_path = path.join(PATH_L, "logits_resnet110_SD_c%i_seed%i" % (num_classes, seed))
        
    evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = file_path, x_val = x_val, y_val = y_val)

