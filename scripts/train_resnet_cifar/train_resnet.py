# Training procedure for CIFAR-10/100 using ResNet 110.
# ResNet model from https://github.com/BIGBALLON/cifar-10-cnn/blob/master/4_Residual_Network/ResNet_keras.py

import keras
import numpy as np
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from sklearn.model_selection import train_test_split
import pickle

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.evaluation import evaluate_model

# Constants
STACK_N            = 18            
IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS       = 3
BATCH_SIZE         = 128
EPOCHS             = 200
ITERATIONS         = 45000 // BATCH_SIZE
WEIGHT_DECAY       = 0.0001
PATH_W = path.join("..", "..", "models")
PATH_L = path.join("..", "..", "logits")

def scheduler(epoch):
    if epoch < 80:
        return 0.1
    if epoch < 150:
        return 0.01
    return 0.001

def residual_network(img_input,classes_num=10,stack_n=5):
    def residual_block(intput,out_channel,increase=False):
        if increase:
            stride = (2,2)
        else:
            stride = (1,1)

        pre_bn   = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(pre_relu)
        bn_1   = BatchNormalization()(conv_1)
        relu1  = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(relu1)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1,1),
                                strides=(2,2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput,conv_2])
        return block

    # build model
    # total layers = STACK_N * 3 * 2 + 2
    # STACK_N = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(STACK_N):
        x = residual_block(x,16,False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,STACK_N):
        x = residual_block(x,32,False)
    
    # input: 16x16x32 output: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,STACK_N):
        x = residual_block(x,64,False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num,activation='softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    return x


# Per channel mean and std normalization
def color_preprocessing(x_train, x_val, x_test):
    
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')    
    x_test = x_test.astype('float32')
    
    # Normalize data with per-pixel mean
    img_mean = x_train.mean(axis=0)  # per-pixel mean
    img_std = x_train.std(axis=0)
    x_train = (x_train-img_mean)/img_std
    x_val = (x_val-img_mean)/img_std
    x_test = (x_test-img_mean)/img_std
    
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
    img_input = Input(shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS))
    output    = residual_network(img_input,num_classes,STACK_N)
    resnet    = Model(img_input, output)
    print(resnet.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    cbks = [LearningRateScheduler(scheduler)]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)

    datagen.fit(x_train45)

    # start training
    hist = resnet.fit_generator(datagen.flow(x_train45, y_train45, batch_size=BATCH_SIZE),
                         steps_per_epoch=ITERATIONS,
                         epochs=EPOCHS,
                         callbacks=cbks,
                         validation_data=(x_val, y_val))
    resnet.save(path.join(PATH_W, 'weights_resnet110_c%i_seed%i.h5' % (num_classes, seed)))
    
    print("Get test accuracy:")
    loss, accuracy = resnet.evaluate(x_test, y_test, verbose=0)
    print("Test: accuracy1 = %f  ;  loss1 = %f" % (accuracy, loss))
    
    print("Pickle models history")
    with open('hist_resnet110_c%i_seed%i.p' % (num_classes, seed), 'wb') as f:
        pickle.dump(hist.history, f)
        

def gen_logits(seed = 13, num_classes = 10):
    
    weights_file = path.join(PATH_W, 'weights_resenet110_c%i_seed%i.h5' % (num_classes, seed))  # TODO check whether weight file exists   
    file_path = path.join(PATH_L, "logits_resenet110_c%i_seed%i" % (num_classes, seed))
    
    (x_train45, y_train45), (x_val, y_val), (x_test, y_test) = prep_data(seed, num_classes)

    
    # build network and evaluate
    img_input = Input(shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS))
    output    = residual_network(img_input,num_classes,STACK_N)
    model2    = Model(img_input, output)    
    evaluate_model(model2, weights_file, x_test, y_test, bins = 15, verbose = True, 
                   pickle_file = file_path, x_val = x_val, y_val = y_val)