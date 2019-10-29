# Calibration methods including Histogram Binning and Temperature Scaling

import numpy as np
from scipy.optimize import minimize 
import pandas as pd
import time
from sklearn.metrics import log_loss, brier_score_loss
from keras.losses import categorical_crossentropy
from os.path import join
import sklearn.metrics as metrics
# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
from utility.unpickle_probs import unpickle_probs
from utility.evaluation import ECE, MCE, Brier, evaluate


from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Lambda
import keras.backend as K
from keras import regularizers
import keras
from keras.utils import to_categorical
from keras.layers import Layer
from keras.layers import Activation

from tensorflow import set_random_seed

from sklearn.linear_model import LogisticRegression


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))  # Subtract max so biggest is 0 to avoid numerical instability
    
    # Axis 0 if only one dimensional array
    axis = 0 if len(e_x.shape) == 1 else 1
    
    return e_x / e_x.sum(axis=axis, keepdims=1)
    

class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    
    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """
    
    def __init__(self, M=15):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.
        
        Params:
            probs: probabilities of data
            true: true labels of data
        """

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf


    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences
        
        Param:
            probs: probabilities of the data (shape [samples, classes])
            
        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
        
        
class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)    
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, logits, true, verbose=False):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        
        if verbose:
            print("Temperature:", 1/self.temp)
        
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)
            
            
class VectorScaling():
    
    def __init__(self, classes = 1, W = [], bias = [], maxiter = 100, solver = "BFGS", use_bias = True):
        """
        Initialize class
        
        Params:
            maxiter (int): maximum iterations done by optimizer.
            classes (int): how many classes in given data set. (based on logits )
            W (np.ndarray): matrix with temperatures for all the classes
            bias ( np.array): vector with biases
        """
        
        self.W = W
        self.bias = bias 
        self.maxiter = maxiter
        self.solver = solver
        self.classes = classes
        self.use_bias = use_bias
    
    def _loss_fun(self, x, logits, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        W = np.diag(x[:self.classes])
        
        if self.use_bias:
            bias = x[self.classes:]
        else:
            bias = np.zeros(self.classes)
        scaled_probs = self.predict(logits, W, bias)    
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        true = true.flatten() # Flatten y_val
        self.classes = logits.shape[1]
        x0 = np.concatenate([np.repeat(1, self.classes), np.repeat(0, self.classes)])
        opt = minimize(self._loss_fun, x0 = x0, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.W = np.diag(opt.x[:logits.shape[1]])
        self.bias = opt.x[logits.shape[1]:]
        
        return opt
        
    def predict(self, logits, W = [], bias = []):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if len(W) == 0 or len(bias) == 0:  # Use class variables
            scaled_logits = np.dot(logits, self.W) + self.bias
        else:  # Take variables W and bias from arguments
            scaled_logits = np.dot(logits, W) + bias
        
        return softmax(scaled_logits)
        
            
            
class MatrixScaling():
    
    def __init__(self, classes = -1, max_epochs = 1000, patience = 5):
        """
        Initialize class
        
        Params:
            max_epochs (int): maximum iterations done by optimizer.
            classes (int): how many classes in given data set. (based on logits )
            patience (int): how many worse epochs before early stopping
        """
        
        if classes >= 1:
            self.model = self.create_model(classes)
        else:
            self.model = None
        self.max_epochs = max_epochs
        self.patience = patience
        self.classes = classes
    
    
    def create_model(self, classes, verbose = True):
        
        model = Sequential()
        model.add(Dense(classes, use_bias=True, input_dim=classes, activation="softmax"))
        
        if verbose:
            model.summary()

        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        
        return model
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal parameters
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the model after minimizing is finished.
        """
            
        self.model = self.create_model(logits.shape[1])
        
        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=0, mode='auto')
        cbs = [early_stop]
        hist = self.model.fit(logits, true, epochs=self.max_epochs, callbacks=cbs)
        
        return hist
        
    def predict(self, logits):
        """
        Scales logits based on the model and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        return self.model.predict(logits)
    
    @property
    def coef_(self):
        if self.model:
            return self.model.get_weights()[0].T
    
    @property
    def intercept_(self):
        if self.model:
            return self.model.get_weights()[1]
        
        

class Dirichlet_NN():
    
    def __init__(self, l2 = 0., mu = None, classes = -1, max_epochs = 500, comp = True,
                 patience = 15, lr = 0.001, weights = [], random_state = 15, loss = "sparse_categorical_crossentropy",
                 double_fit = True, use_logits = False):
        """
        Initialize class
        
        Params:
            l2 (float): regularization for off-diag regularization.
            mu (float): regularization for bias. (if None, then it is set equal to lambda of L2)
            classes (int): how many classes in given data set. (based on logits)
            max_epochs (int): maximum iterations done by optimizer.
            comp (bool): whether use complementary (off_diag) regularization or not.
            patience (int): how many worse epochs before early stopping
            lr (float): learning rate of Adam optimizer
            weights (array): initial weights of model ([k,k], [k]) - weights + bias
            random_state (int): random seed for numpy and tensorflow
            loss (string/class): loss function to optimize
            double_fit (bool): fit twice the model, in the beginning with lr (default=0.001), and the second time 10x lower lr (lr/10)
            use_logits (bool): Using logits as input of model, leave out conversion to logarithmic scale.

        """
        
        if classes >= 1:
            self.model = self.create_model(classes, weights)
        else:
            self.model = None
        self.max_epochs = max_epochs
        self.patience = patience
        self.classes = classes
        self.l2 = l2
        self.lr = lr
        self.weights = weights
        self.random_state = random_state
        self.loss = loss
        self.double_fit = double_fit
        self.use_logits = use_logits
        
        if mu:
            self.mu = mu
        else:
            self.mu = l2
            
        if comp:
            self.regularizer = self.L2_offdiag(l2 = self.l2)
        else:
            self.regularizer = keras.regularizers.l2(l = self.l2)
            
        set_random_seed(random_state)
        np.random.seed(random_state)    
    
    def create_model(self, classes, weights=[], verbose = False):
    
        """
        Create model and add loss to it
        
        Params:
            classes (int): number of classes, used for input layer shape and output shape
            weights (array): starting weights in shape of ([k,k], [k]), (weights, bias)
            verbose (bool): whether to print out anything or not
        
        Returns:
            model (object): Keras model
        """
        
        model = Sequential()
        if not self.use_logits: # Leave out converting to logarithmic scale if logits are used as input.
            model.add(Lambda(self._logFunc, input_shape=[classes]))
            
            model.add(Dense(classes, activation="softmax"
                        , kernel_initializer=keras.initializers.Identity(gain=1)
                        , bias_initializer="zeros",
                kernel_regularizer=self.regularizer, bias_regularizer=keras.regularizers.l2(l=self.mu)))
            
        else:
            model.add(Dense(classes, input_shape=[classes], activation="softmax"
                        , kernel_initializer=keras.initializers.Identity(gain=1)
                        , bias_initializer="zeros",
                kernel_regularizer=self.regularizer, bias_regularizer=keras.regularizers.l2(l=self.mu)))
        

            
        if len(weights) != 0:  # Weights that are set from fitting
            model.set_weights(weights)
        elif len(self.weights) != 0:  # Weights that are given from initialisation
            model.set_weights(self.weights)

        
        adam = keras.optimizers.Adam(lr=self.lr)
        model.compile(loss=self.loss, optimizer=adam)
        
        if verbose:
            model.summary()
        
        return model
    
    def fit(self, probs, true, weights = [], verbose = False, double_fit = None, batch_size = 128):
        """
        Trains the model and finds optimal parameters
        
        Params:
            probs: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            weights (array): starting weights in shape of ([k,k], [k]), (weights, bias)
            verbose (bool): whether to print out anything or not
            double_fit (bool): fit twice the model, in the beginning with lr (default=0.001), and the second time 10x lower lr (lr/10)
            
        Returns:
            hist: Keras history of learning process
        """
        
        if len(weights) != 0:
            self.weights = weights
            
        if "sparse" not in self.loss:  # Check if need to make Y categorical; TODO Make it more see-through
            true = to_categorical(true)
            
        if double_fit == None:
            double_fit = self.double_fit
            
        self.model = self.create_model(probs.shape[1], self.weights, verbose)
        
        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=verbose, mode='auto')
        cbs = [early_stop]
        
        hist = self.model.fit(probs, true, epochs=self.max_epochs, callbacks=cbs, batch_size=batch_size, verbose=verbose)
        
        if double_fit:  # In case of my experiments it gave better results to start with default learning rate (0.001) and then fit again (0.0001) learning rate.
            if verbose:
                print("Fit with 10x smaller learning rate")
            self.lr = self.lr/10
            self.fit(probs, true, weights = self.model.get_weights(), verbose=verbose, double_fit=False, batch_size = batch_size)  # Fit 2 times
        
        return hist
        
    def predict(self, probs):  # TODO change it to return only the best prediction
        """
        Scales logits based on the model and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        return self.model.predict(probs)
        
    
    def predict_proba(self, probs):
        """
        Scales logits based on the model and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        return self.model.predict(probs)
    
    @property
    def coef_(self):
        """
        Actually weights of neurons, but to keep similar notation to original Dirichlet we name it coef_
        """
        if self.model:
            return self.model.get_weights()[0].T  # Transposed to match with full dirichlet weights.
    
    @property
    def intercept_(self):
        """
        Actually bias values, but to keep similar notation to original Dirichlet we name it intercept_
        """
        if self.model:
            return self.model.get_weights()[1]
        
        
        
    def _logFunc(self, x):
        """
        Find logarith of x (tensor)
        """
        eps = np.finfo(float).eps  # 1e-16
        
        return K.log(K.clip(x, eps, 1 - eps)) # How this clip works? K.clip(x, K.epsilon(), None) + 1.)

    
    # Inner classes for off diagonal regularization
    class Regularizer(object):
        """
        Regularizer base class.
        """

        def __call__(self, x):
            return 0.0

        @classmethod
        def from_config(cls, config):
            return cls(**config)


    class L2_offdiag(Regularizer):
        """
        Regularizer for L2 regularization off diagonal.
        """

        def __init__(self, l2=0.0):
        
            """
            Params:
                l: (float) lambda, L2 regularization factor.
            """
            self.l2 = K.cast_to_floatx(l2)

        def __call__(self, x):
            """
            Off-diagonal regularization (complementary regularization)
            """

            reg = 0

            for i in range(0, x.shape[0]):
                reg += K.sum(self.l2 * K.square(x[0:i, i]))
                reg += K.sum(self.l2 * K.square(x[i+1:, i]))
                
            return reg

        def get_config(self):
            return {'l2': float(self.l2)}
            
            
class Dirichlet_diag_NN():
    
    def __init__(self, classes = -1, max_epochs = 500, patience = 15, random_state = 15):
        """
        Initialize class
        
        Params:
            max_epochs (int): maximum iterations done by optimizer.
            classes (int): how many classes in given data set. (based on logits )
            patience (int): how many worse epochs before early stopping
        """
        
        if classes >= 1:
            self.model = self.create_model(classes)
        else:
            self.model = None
        self.max_epochs = max_epochs
        self.patience = patience
        self.classes = classes
        
        set_random_seed(random_state)
        np.random.seed(random_state)
    
    
    def create_model(self, classes, verbose = True):
        
        model = Sequential()
        model.add(Lambda(self._logFunc, input_shape=[classes]))
        model.add(DiagonalLayer(output_dim=classes))
        model.add(Activation('softmax'))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        
        if verbose:
            model.summary()
        
        return model
    
    # Find the temperature
    def fit(self, logits, true, verbose = True, batch_size = 128):
        """
        Trains the model and finds optimal parameters
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the model after minimizing is finished.
        """
            
        self.model = self.create_model(logits.shape[1], verbose)
        
        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=0, mode='auto')
        cbs = [early_stop]
                
        hist = self.model.fit(logits, true, epochs=self.max_epochs, callbacks=cbs, verbose = verbose, batch_size = batch_size)
        
        return hist
        
    def predict(self, logits):
        """
        Scales logits based on the model and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        return self.model.predict(logits)
    
    @property
    def coef_(self):
        if self.model:
            return self.model.get_weights()[0].T
    
    @property
    def intercept_(self):
        if self.model:
            return self.model.get_weights()[1]
            
    
    def _logFunc(self, x):
        """
        Find logarith of x (tensor)
        """
        eps = np.finfo(float).eps  # 1e-16
        
        return K.log(K.clip(x, eps, 1 - eps)) # How this clip works? K.clip(x, K.epsilon(), None) + 1.)
            
            
            
class VectorScaling_NN():
    
    def __init__(self, classes = -1, max_epochs = 500, patience = 15, random_state = 15):
        """
        Initialize class
        
        Params:
            max_epochs (int): maximum iterations done by optimizer.
            classes (int): how many classes in given data set. (based on logits )
            patience (int): how many worse epochs before early stopping
        """
        
        if classes >= 1:
            self.model = self.create_model(classes)
        else:
            self.model = None
        self.max_epochs = max_epochs
        self.patience = patience
        self.classes = classes
        
        set_random_seed(random_state)
        np.random.seed(random_state)
    
    
    def create_model(self, classes, verbose = True):
        
        model = Sequential()
        model.add(DiagonalLayer(input_shape=[classes], output_dim=classes))
        model.add(Activation('softmax'))
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        
        if verbose:
            model.summary()
        
        return model
    
    # Find the temperature
    def fit(self, logits, true, verbose = True, batch_size = 128):
        """
        Trains the model and finds optimal parameters
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the model after minimizing is finished.
        """
            
        self.model = self.create_model(logits.shape[1], verbose)
        
        early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=self.patience, verbose=0, mode='auto')
        cbs = [early_stop]
                
        hist = self.model.fit(logits, true, epochs=self.max_epochs, callbacks=cbs, verbose = verbose, batch_size = batch_size)
        
        return hist
        
    def predict(self, logits):
        """
        Scales logits based on the model and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        return self.model.predict(logits)
    
    @property
    def coef_(self):
        if self.model:
            return self.model.get_weights()[0].T
    
    @property
    def intercept_(self):
        if self.model:
            return self.model.get_weights()[1]
            


class DiagonalLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DiagonalLayer, self).__init__(**kwargs)

    def build(self, input_shape, activation="softmax"):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=[self.output_dim],
                                      initializer="ones",
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=[self.output_dim],
                                    initializer='zeros',
                                    trainable=True)
        super(DiagonalLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.np.multiply(x, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
            
   

def log_encode(x):
    eps = np.finfo(x.dtype).eps
    x = np.clip(x, eps, 1)
    return np.log(x)
    


class LogisticCalibration(LogisticRegression):
    def __init__(self, C=1.0, solver='lbfgs', multi_class='multinomial',
                 log_transform=True):
        self.C_grid = C
        self.C = C if isinstance(C, float) else C[0]
        self.solver = solver
        self.log_transform = log_transform
        self.encode = log_encode
        self.multiclass=multi_class
        super(LogisticCalibration, self).__init__(C=C, solver=solver,
                                                  multi_class=multi_class)

    def fit(self, scores, y, X_val=None, y_val=None, *args, **kwargs):
        if isinstance(self.C_grid, list):
            calibrators = []
            losses = np.zeros(len(self.C_grid))
            for i, C in enumerate(self.C_grid):
                cal = LogisticCalibration(C=C, solver=self.solver,
                                          multi_class=self.multi_class,
                                          log_transform=self.log_transform)
                cal.fit(scores, y)
                losses[i] = log_loss(y_val, cal.predict_proba(X_val))
                calibrators.append(cal)
            best_idx = losses.argmin()
            self.C = calibrators[best_idx].C
        return super(LogisticCalibration, self).fit(self.encode(scores), y,
                *args, **kwargs)

    def predict_proba(self, scores, *args, **kwargs):
        return super(LogisticCalibration,
                self).predict_proba(self.encode(scores), *args, **kwargs)

    def predict(self, scores, *args, **kwargs):
        return super(LogisticCalibration, self).predict(self.encode(scores),
                *args, **kwargs)   

    

# TODO mode this ending part to other file
from sklearn.preprocessing import OneHotEncoder

def get_preds_all(y_probs, y_true, axis = 1, normalize = False, flatten = True):
    
    y_preds = np.argmax(y_probs, axis=axis)  # Take maximum confidence as prediction
    y_preds = y_preds.reshape(-1, 1)
    
    if normalize:
        y_probs /= np.sum(y_probs, axis=axis)
        
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)   
    enc.fit(y_preds)

    y_preds = enc.transform(y_preds)
    y_true = enc.transform(y_true)
    
    if flatten:        
        y_preds = y_preds.flatten()
        y_true = y_true.flatten()
        y_probs = y_probs.flatten()
        
    return y_preds, y_probs, y_true
    
    
def evaluate_legacy(probs, y_true, verbose = False, normalize = True, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
        
    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    
    if normalize:
        confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence
    
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
    # get predictions, confidences and true labels for all classes  
    preds2, confs2, y_true2 = get_preds_all(probs, y_true, normalize=False, flatten=True)

    # Calculate ECE and ECE2
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    ece2 = ECE(confs2, preds2, y_true2, bin_size = 1/bins, ece_full=True)
    
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size = 1/bins)
    mce2 = MCE(confs2, preds2, y_true2, bin_size = 1/bins, ece_full=True)

    loss = log_loss(y_true=y_true, y_pred=probs)
    
    #y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    #brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE), NB! not correct
    
    brier = Brier(probs, y_true)
    
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("ECE2:", ece2)
        print("MCE:", mce)
        print("MCE2:", mce2)
        print("Loss:", loss)
        print("brier:", brier)
    
    return (error, ece, ece2, mce, mce2, loss, brier)
    
 
def cal_results(fn, path, files, m_kwargs = {}, approach = "all", input = "probabilities"):
    
    """
    Calibrate models scores, using output from logits files and given function (fn). 
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.
    
    TODO: split calibration of single and all into separate functions for more use cases.
    
    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict", 
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        m_kwargs (dictionary): keyword arguments for the calibration class initialization
        approach (string): "all" for multiclass calibration and "1-vs-K" for 1-vs-K approach.
        input (string): "probabilities" or "logits", specific to calibration method
        
    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.
    
    """
    
    df = pd.DataFrame(columns=["Name", "Error", "ECE", "ECE2", "ECE_CW", "ECE_CW2", "ECE_FULL", "ECE_FULL2", "MCE", "MCE2", "Loss", "Brier"])
    
    total_t1 = time.time()
   
    
    for i, f in enumerate(files):
    
        
        name = "_".join(f.split("_")[1:-1])
        print(name)
        t1 = time.time()

        FILE_PATH = join(path, f)
        (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)
        
        # Specify the input type, some of the calibration methods need logits as inputs, others probabilities
        if input == "probabilities":
            input_val = softmax(logits_val)  # Softmax logits
            input_test = softmax(logits_test)
        else:
            input_val = logits_val
            input_test = logits_test
            
        
        # Train and test model based on the approach "all" or "1-vs-K"
        if approach == "all":            

            y_val_flat = y_val.flatten()

            model = fn(**m_kwargs)

            opt = model.fit(input_val, y_val_flat)
            

            probs_val = model.predict(input_val) 
            probs_test = model.predict(input_test)
            
            error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate(softmax(logits_test), y_test, verbose=False)  # Uncalibrated results
            error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2, brier2 = evaluate(probs_test, y_test, verbose=False)
            error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3, brier3 = evaluate(probs_val, y_val, verbose=False)

            print("Uncal Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f, brier %f" % (error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))
            print("Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f, brier %f" % (error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2, brier2))
            print("Validation Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f, brier %f" % (error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3, brier3))
            
            
        else:  # 1-vs-k models

            K = input_test.shape[1]
            
            probs_val = np.zeros_like(input_val)
            probs_test = np.zeros_like(input_test)
            
            # Go through all the classes
            for k in range(K):
                # Prep class labels (1 fixed true class, 0 other classes)
                y_cal = np.array(y_val == k, dtype="int")[:, 0]

                # Train model
                model = fn(**m_kwargs)
                model.fit(input_val[:, k], y_cal) # Get only one column with probs for given class "k"
                


                probs_val[:, k] = model.predict(input_val[:, k])  # Predict new values based on the fittting
                probs_test[:, k] = model.predict(input_test[:, k])

                # Replace NaN with 0, as it should be close to zero  # TODO is it needed?
                idx_nan = np.where(np.isnan(probs_test))
                probs_test[idx_nan] = 0

                idx_nan = np.where(np.isnan(probs_val))
                probs_val[idx_nan] = 0

            error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate(softmax(logits_test), y_test, verbose=False)  # Uncalibrated results
            error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2, brier2 = evaluate(probs_test, y_test, verbose=False)
            error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3, brier3 = evaluate(probs_val, y_val, verbose=False)

            print("Uncal Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f, brier %f" % (error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))
            print("Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f, brier %f" % (error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2, brier2))
            print("Validation Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f, brier %f" % (error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3, brier3))
            
            
            
        
        df.loc[i*3] = [name, error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier]
        df.loc[i*3+1] = [(name + "_calib"), error2, ece2_1, ece2_2, ece_cw2_1, ece_cw2_2, ece_full2_1, ece_full2_2, mce2_1, mce2_2, loss2, brier2]
        
        df.loc[i*3+2] = [(name + "_val_calib"), error3, ece3_1, ece3_2, ece_cw3_1, ece_cw3_2, ece_full3_1, ece_full3_2, mce3_1, mce3_2, loss3, brier3]
        
        t2 = time.time()
        print("Time taken:", (t2-t1), "\n")
        
    total_t2 = time.time()
    print("Total time taken:", (total_t2-total_t1))
        
    return df
    