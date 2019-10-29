# Training procedure for LeNet-5 CIFAR-10/100.
#Code base from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/1_Lecun_Network/LeNet_dp_da_wd_keras.py 

import keras
import numpy as np
from keras import optimizers
from keras.datasets import cifar10, cifar100
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model

BATCH_SIZE    = 128
EPOCHS        = 300
ITERATIONS    = 45000 // BATCH_SIZE
WEIGHT_DECAY  = 0.0001
N = 1
PATH_W = path.join("..", "..", "models")
PATH_L = path.join("..", "..", "logits")

def build_model(n=1, num_classes = 10):
    """
    parameters:
        n: (int) scaling for model (n times filters in Conv2D and nodes in Dense)
    """
    model = Sequential()
    model.add(Conv2D(n*6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(WEIGHT_DECAY), input_shape=(32,32,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization(epsilon=1.1e-5))
    model.add(Conv2D(n*16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(WEIGHT_DECAY)))
    model.add(BatchNormalization(epsilon=1.1e-5))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(n*120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(WEIGHT_DECAY) ))
    model.add(Dense(n*84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(WEIGHT_DECAY) ))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(WEIGHT_DECAY) ))
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler10(epoch):
    if epoch <= 60:
        return 0.1
    if epoch <= 120:
        return 0.01
    if epoch <= 160:    
        return 0.001
    return 0.0001
    
def scheduler100(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:    
        return 0.002
    return 0.0004
    
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

    # build network
    model = build_model(n=N, num_classes = num_classes)
    print(model.summary())
    
    if num_classes == 10:
        scheduler = scheduler10
    else:
        scheduler = scheduler100

    # set callback
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr]

    # using real-time data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
            

    datagen.fit(x_train45)

    # start traing 
    hist = model.fit_generator(datagen.flow(x_train45, y_train45,batch_size=BATCH_SIZE, shuffle=True),
                        steps_per_epoch=ITERATIONS,
                        epochs=EPOCHS,
                        callbacks=cbks,
                        validation_data=(x_val, y_val))
    # save model
    model.save(path.join(PATH_W, 'weights_lenet_c%i_seed%i.h5' % (num_classes, seed)))
    
    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))

    print("Pickle models history")
    with open('hist_lenet_c%i_seed%i.p' % (num_classes, seed), 'wb') as f:
        pickle.dump(hist.history, f)
        
        
def gen_logits(seed = 13, num_classes = 10):

    print("Change v1")

    weights_file = path.join(PATH_W, 'weights_lenet_c%i_seed%i.h5' % (num_classes, seed))  # TODO check whether weight file exists   
    file_path = path.join(PATH_L, "logits_lenet_c%i_seed%i" % (num_classes, seed))

    print(weights_file)
    (x_train45, y_train45), (x_val, y_val), (x_test, y_test) = prep_data(seed, num_classes)
    
    # build network
    model = build_model(n = N, num_classes = num_classes)
    evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = file_path, x_val = x_val, y_val = y_val)