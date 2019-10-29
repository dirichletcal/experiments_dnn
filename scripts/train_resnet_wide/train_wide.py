# Training procedure for WideNet 32-10 for CIFAR-10 and CIFAR-100.

import keras
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers
import train_resnet_wide.wide_residual_network as wrn
from keras import regularizers
from sklearn.model_selection import train_test_split
import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model

depth              = 34  # 32, if ignoring conv layers carrying residuals, which are needed for increasing filter size.
growth_rate        = 10  # Growth factor
n                  = (depth-4)//6
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = 128
epochs             = 200
iterations         = 45000 // batch_size
weight_decay       = 0.0005
PATH_W = path.join("..", "..", "models")
PATH_L = path.join("..", "..", "logits")

def scheduler(epoch):
    if epoch <= 60:
        return 0.1
    if epoch <= 120:
        return 0.02
    if epoch <= 160:
        return 0.004
    return 0.0008

    
# Preprocessing based on the paper http://arxiv.org/abs/1605.07146
# and their code https://github.com/szagoruyko/wide-residual-networks
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
    

# Main method
def train(seed = 13, num_classes = 10):


    (x_train45, y_train45), (x_val, y_val), (x_test, y_test) = prep_data(seed, num_classes)

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))    
    model = wrn.create_wide_residual_network(img_input, nb_classes=num_classes, N=n, k=growth_rate, dropout=0.0)
    print(model.summary())
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    change_lr = LearningRateScheduler(scheduler)
    checkpointer = ModelCheckpoint(path.join(PATH_W, 'model_wide_28_10_c%i_best.hdf5' % num_classes), verbose=1, save_best_only=True)
    cbks = [change_lr, checkpointer]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect') # Missing pixels replaced with reflections

    datagen.fit(x_train45)

    # start training
    hist = model.fit_generator(datagen.flow(x_train45, y_train45, batch_size=batch_size, shuffle=True),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_val, y_val))
    model.save(path.join(PATH_W, 'weights_resnet_wide32_c%i.h5' % num_classes))
    
    print("Get test accuracy:")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_wide32_cifar%i.p' % num_classes, 'wb') as f:
        pickle.dump(hist.history, f)
        

def gen_logits(seed = 13, num_classes = 10):

    weights_file = path.join(PATH_W, 'weights_resnet_wide32_c%i.h5' % num_classes)  # TODO check whether weight file exists

    (x_train45, y_train45), (x_val, y_val), (x_test, y_test) = prep_data(seed, num_classes)

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))    
    model = wrn.create_wide_residual_network(img_input, nb_classes=num_classes, N=n, k=growth_rate, dropout=0.0)
    # set optimizer
    
    file_path = path.join(PATH_L, "logits_resnet_wide32_c%i_seed%i" % (num_classes, seed))
    
    evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = file_path, x_val = x_val, y_val = y_val)