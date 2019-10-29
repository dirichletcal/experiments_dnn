from keras.datasets import cifar10
import numpy as np

def load_cifar_n_classes(classes = [1,3,5]):  # Default car, cat, dog
        
    (x_train, y_train), (x_test, y_test) = cifar10.load_data() # Load in CIFAR-10 dataset
    
    indices_tr = [idx for idx in range(len(y_train)) if y_train[idx] in classes] # Get indices of certain samples
    indices_te = [idx for idx in range(len(y_test)) if y_test[idx] in classes]  # Get indices of certain samples
    
    y_train = y_train[indices_tr]
    y_test = y_test[indices_te]
    
    # Shift class labels to 0,1,2
    for i, y in enumerate(y_train):
        y_train[i] = np.where(classes == y)[0]
        
    for i, y in enumerate(y_test):
        y_test[i] = np.where(classes == y)[0]

    
    return (x_train[indices_tr], y_train), (x_test[indices_te], y_test)  # Return samples from correct classes