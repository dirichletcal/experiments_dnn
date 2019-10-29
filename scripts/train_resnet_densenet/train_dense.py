# Training procedure for CIFAR-100 using DenseNet 40 with growth rate 12.


from __future__ import print_function

from train_resnet_densenet import densenet
import numpy as np
import sklearn.metrics as metrics
import keras
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from sklearn.model_selection import train_test_split
import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model

BATCH_SIZE = 64
EPOCHS = 300

IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS = 3

IMG_DIM = (IMG_CHANNELS, IMG_ROWS, IMG_COLS) if K.image_dim_ordering() == "th" else (IMG_ROWS, IMG_COLS, IMG_CHANNELS)
DEPTH = 40
DENSE_BLOCK = 3
GROWTH_RATE = 12
DROPOUT_RATE = 0.0 # 0.0 for data augmentation
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.1

PATH_W = path.join("..", "..", "models")
PATH_L = path.join("..", "..", "logits")

def scheduler(epoch):
    if epoch < EPOCHS/2:
        return LEARNING_RATE
    elif epoch < EPOCHS*3/4:
        return LEARNING_RATE*0.1
    return LEARNING_RATE*0.01

# Preprocessing for DenseNet https://arxiv.org/pdf/1608.06993v3.pdf
def color_preprocessing(x_train, x_val, x_test):
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_val[:,:,:,i] = (x_val[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

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
        width_shift_range=0.125,  # 0.125*32 = 4 so max padding of 4 pixels, as described in paper.
        height_shift_range=0.125,  # first zero-padded with 4 pixels on each side, then randomly cropped to again produce 32×32 images
        fill_mode = "constant",
        cval = 0
    )

    img_gen.fit(x_train45, seed=seed)

    callbacks = [LearningRateScheduler(scheduler)]
    
    if num_classes == 10:
        nb_filter = -1
    else:
        nb_filter = 12


    model = densenet.DenseNet(IMG_DIM, classes=num_classes, depth=DEPTH, nb_dense_block=DENSE_BLOCK,
                              growth_rate=GROWTH_RATE, nb_filter=nb_filter, dropout_rate=DROPOUT_RATE, weights=None, weight_decay = WEIGHT_DECAY)
    print("Model created")

    model.summary()
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=True)  # dampening = 0.9? Should be zero?
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    hist = model.fit_generator(img_gen.flow(x_train45, y_train45, batch_size=BATCH_SIZE, shuffle=True),
                        steps_per_epoch=len(x_train45) // BATCH_SIZE, epochs=EPOCHS,
                        callbacks=callbacks,
                        validation_data=(x_val, y_val),
                        validation_steps=x_val.shape[0] // BATCH_SIZE, verbose=1)
                        

    model.save(path.join(PATH_W, 'weights_densenet40_c%i_seed%i.h5' % (num_classes, seed)))

    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

    print("Pickle models history")
    with open('hist_densenet40_c%i_seed%i.p' % (num_classes, seed), 'wb') as f:
        pickle.dump(hist.history, f)
    



def gen_logits(seed = 13, num_classes = 13):

    weights_file = path.join(PATH_W, 'weights_densenet40_c%i_seed%i.h5' % (num_classes, seed))  # TODO check whether weight file exists   
    file_path = path.join(PATH_L, "logits_densenet40_c%i_seed%i" % (num_classes, seed))

    (x_train45, y_train45), (x_val, y_val), (x_test, y_test) = prep_data(seed, num_classes)
    
    if num_classes == 10:
        nb_filter = -1
    else:
        nb_filter = 12
    
    model = densenet.DenseNet(IMG_DIM, classes=num_classes, depth=DEPTH, nb_dense_block=DENSE_BLOCK,
                              growth_rate=GROWTH_RATE, nb_filter=nb_filter, dropout_rate=DROPOUT_RATE, weights=None, weight_decay=WEIGHT_DECAY)

    evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = file_path, x_val = x_val, y_val = y_val)