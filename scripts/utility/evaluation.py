# Functions for calibration of results
from __future__ import division, print_function
import sklearn.metrics as metrics
import numpy as np
import pickle
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
import sklearn.metrics as metrics
from scipy.stats import percentileofscore
from sklearn.preprocessing import label_binarize

    
def evaluate_model(model, weights_file, x_test, y_test, bins = 15, verbose = True, pickle_file = None, x_val = None, y_val = None):
    """
    Evaluates the model, in addition calculates the calibration errors and 
    saves the logits for later use, if "pickle_file" is not None.
    
    Parameters:
        model (keras.model): constructed model
        weights (string): path to weights file
        x_test: (numpy.ndarray) with test data
        y_test: (numpy.ndarray) with test data labels
        verbose: (boolean) print out results or just return these
        pickle_file: (string) path to pickle probabilities given by model
        x_val: (numpy.ndarray) with validation data
        y_val: (numpy.ndarray) with validation data labels

        
    Returns:
        (acc, ece, mce): accuracy of model, ECE and MCE (calibration errors)
    """
    
    # Change last activation to linear (instead of softmax)
    last_layer = model.layers.pop()
    last_layer.activation = keras.activations.linear
    i = model.input
    o = last_layer(model.layers[-1].output)
    model = keras.models.Model(inputs=i, outputs=[o])
    
    # First load in the weights
    model.load_weights(weights_file)
    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    
    # Next get predictions
    logits = model.predict(x_test, verbose=1)
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    
    # Find accuracy and error
    if y_test.shape[1] > 1:  # If 1-hot representation, get back to numeric   
        y_test = np.array([[np.where(r==1)[0][0]] for r in y_test]) # Back to np array also
       
    accuracy = metrics.accuracy_score(y_test, preds) * 100
    error = 100 - accuracy
    
    # Confidence of prediction    
    ece = ECE(probs, y_test, bin_size = 1/bins)
    ece_cw = classwise_ECE(probs, y_test, bins = bins, power = 1)
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", ece_cw)
        
     
    # Pickle probabilities for test and validation
    if pickle_file:
    
        #Get predictions also for x_val
        logits_val = model.predict(x_val)
        probs_val = softmax(logits_val)
        preds_val = np.argmax(probs_val, axis=1)
        
        # 
        if y_val.shape[1] > 1:  # If 1-hot representation, get back to numeric   
            y_val = np.array([[np.where(r==1)[0][0]] for r in y_val])  # Also convert back to np.array, TODO argmax?
            
        if verbose:
            print("Pickling the probabilities for validation and test.")
            print("Validation accuracy: ", metrics.accuracy_score(y_val, preds_val) * 100)
            
        # Write file with pickled data
        with open(pickle_file + '.p', 'wb') as f:
            pickle.dump([(logits_val, y_val),(logits, y_test)], f)
    
    # Return the basic results
    return (accuracy, ece, ece_cw)
    
    
def evaluate(probs, y_true, verbose = False, normalize = True, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, NLL, Brier Score
    
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
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
    # Calculate ECE and ECE2, + Classwise and Full (ECE2 =? Full_ECE)
    ece = ECE(probs, y_true, bin_size = 1/bins)
    ece2 = ECE(probs, y_true, bin_size = 1/bins, ece_full=True, normalize = normalize)
    ece_cw = classwise_ECE(probs, y_true, bins = bins, power = 1)
    ece_full = full_ECE(probs, y_true, bins = bins, power = 1)
    
    ece_cw2 = classwise_ECE(probs, y_true, bins = bins, power = 2)
    ece_full2 = full_ECE(probs, y_true, bins = bins, power = 2)
    
    # Calculate MCE
    mce = MCE(probs, y_true, bin_size = 1/bins, normalize = normalize)
    mce2 = MCE(probs, y_true, bin_size = 1/bins, ece_full=True, normalize = normalize)

    loss = log_loss(y_true=y_true, y_pred=probs)
    
    #y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    #brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE), NB! not correct
    
    brier = Brier(probs, y_true)
    
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("ECE2:", ece2)
        print("ECE_CW", ece_cw)
        print("ECE_CW2", ece_cw)
        print("ECE_FULL", ece_full)
        print("ECE_FULL2", ece_full2)
        print("MCE:", mce)
        print("MCE2:", mce2)
        print("Loss:", loss)
        print("brier:", brier)
    
    return (error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier)


def evaluate_rip(probs, y_true, verbose = False, normalize = True, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, NLL, Brier Score
    
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
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
    # Calculate ECE and ECE2, + Classwise and Full (ECE2 =? Full_ECE)
    ece = ECE(probs, y_true, bin_size = 1/bins)
    ece2 = -1
    ece_cw = classwise_ECE(probs, y_true, bins = bins, power = 1)
    ece_full = -1
    
    ece_cw2 = -1
    ece_full2 = -1
    
    # Calculate MCE
    mce = MCE(probs, y_true, bin_size = 1/bins, normalize = normalize)
    mce2 = -1

    loss = log_loss(y_true=y_true, y_pred=probs)
    
    #y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    #brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE), NB! not correct
    
    brier = Brier(probs, y_true)
    
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("ECE2:", ece2)
        print("ECE_CW", ece_cw)
        print("ECE_CW2", ece_cw)
        print("ECE_FULL", ece_full)
        print("ECE_FULL2", ece_full2)
        print("MCE:", mce)
        print("MCE2:", mce2)
        print("Loss:", loss)
        print("brier:", brier)
    
    return (error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier)

def evaluate_slim(probs, y_true, verbose = False, normalize = True, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, NLL, Brier Score
    
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
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
    # Calculate ECE and ECE2, + Classwise and Full (ECE2 =? Full_ECE)
    ece = ECE(probs, y_true, bin_size = 1/bins)
    ece_cw = classwise_ECE(probs, y_true, bins = bins, power = 1)
    
    # Calculate MCE
    mce = MCE(probs, y_true, bin_size = 1/bins, normalize = normalize)

    loss = log_loss(y_true=y_true, y_pred=probs)
    
    #y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    #brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE), NB! not correct
    
    brier = Brier(probs, y_true)
    
    
    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("ECE_CW", ece_cw)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)
    
    return (error, ece, ece_cw, mce, loss, brier)

def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(x - np.max(x))  # Subtract max, so the biggest is 0 to avoid numerical instability
    
    # Axis 0 if only one dimensional array
    axis = 0 if len(e_x.shape) == 1 else 1
    
    return e_x / e_x.sum(axis=axis, keepdims=1)
    
    
def get_preds_all(y_probs, y_true, axis = 1, normalize = False, flatten = True):
    
    y_preds = np.argmax(y_probs, axis=axis)  # Take maximum confidence as prediction
    y_preds = y_preds.reshape(-1, 1)
    
    if normalize:
        y_probs /= np.sum(y_probs, axis=axis).reshape(-1,1)
        
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)   
    enc.fit(y_preds)

    y_preds = enc.transform(y_preds)
    y_true = enc.transform(y_true)
    
    if flatten:        
        y_preds = y_preds.flatten()
        y_true = y_true.flatten()
        y_probs = y_probs.flatten()
        
    return y_preds, y_probs, y_true
    

def compute_acc_bin_legacy(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin
        

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true, ece_full = False):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        pred_thresh (float) : float in range (0,1), indicating the prediction threshold
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if (x[2] > conf_thresh_lower or conf_thresh_lower == 0) and x[2] <= conf_thresh_upper]  
    
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        if ece_full:
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples])/len_bin  # Avg confidence of BIN
            accuracy = np.mean([x[1] for x in filtered_tuples])  # Mean difference from actual class
        
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
            accuracy = float(correct)/len_bin  # accuracy of BIN
        
    return accuracy, avg_conf, len_bin
  

def ECE(probs, true, bin_size = 0.1, ece_full = False, normalize = False):

    """
    Expected Calibration Error
    
    Args:
        probs (numpy.ndarray): list of probabilities (samples, nr_classes)
        true (numpy.ndarray): list of true labels (samples, 1)
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    
    probs = np.array(probs)
    true = np.array(true)
    
    if len(true.shape) == 2 and true.shape[1] > 1:
        true = true.argmax(axis=1).reshape(-1, 1)
    
    if ece_full:
    
        pred, conf, true = get_preds_all(probs, true, normalize=normalize, flatten=ece_full)
    else:
        pred = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

        if normalize:
            conf = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            conf = np.max(probs, axis=1)  # Take only maximum confidence

    
    # get predictions, confidences and true labels for all classes  
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true, ece_full)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
        
      
def MCE(probs, true, bin_size = 0.1, ece_full=False, normalize = False):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    if ece_full:
    
        pred, conf, true = get_preds_all(probs, true, normalize=normalize, flatten=ece_full)
    else:
        pred = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

        if normalize:
            conf = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            conf = np.max(probs, axis=1)  # Take only maximum confidence
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true, ece_full)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)
    
    
def Brier(probs, true):

    """
    Brier score (mean squared error)
    
    Args:
        probs (list): 2-D list of probabilities
        true (list): 1-D list of true labels
        
    Returns:
        brier: brier score
    """
    
    assert len(probs) == len(true)
    
    n = len(true)  # number of samples
    k = len(probs[0])  # number of classes
    
    brier = 0
     
    for i in range(n):  # Go through all the samples
        for j in range(k):  # Go through all the classes
            y = 1 if j == true[i] else 0  # Check if correct class
            brier += (probs[i][j] - y)**2  # squared error
            
    return brier/n/k  # Mean squared error (should also normalize by number of classes?)

def get_bin_info(conf, pred, true, bin_size = 0.1):

    """
    Get accuracy, confidence and elements in bin information for all the bins.
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    accuracies = []
    confidences = []
    bin_lengths = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)
        
        
    return accuracies, confidences, bin_lengths
    
    
    
def binary_ECE(probs, y_true, power = 1, bins = 15):

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
    bin_func = lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power) * np.sum(idx) / len(probs)

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece

def classwise_ECE(probs, y_true, power = 1, bins = 15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))
    
    n_classes = probs.shape[1]

    return np.sum(
        [
            binary_ECE(
                probs[:, c], y_true[:, c].astype(float), power = power, bins = bins
            ) for c in range(n_classes)
        ]
    )


def simplex_binning(probs, y_true, bins = 15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))
    
    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

    prob_bins = {}
    label_bins = {}

    for i, row in enumerate(idx):
        try:
           prob_bins[','.join([str(r) for r in row])].append(probs[i])
           label_bins[','.join([str(r) for r in row])].append(y_true[i])
        except KeyError:
           prob_bins[','.join([str(r) for r in row])] = [probs[i]]
           label_bins[','.join([str(r) for r in row])] = [y_true[i]]
    
    bins = []
    for key in prob_bins:
        bins.append(
            [
                len(prob_bins[key]),
                np.mean(np.array(prob_bins[key]), axis=0),
                np.mean(np.array(label_bins[key]), axis=0)
            ]
        )

    return bins


def full_ECE(probs, y_true, bins = 15, power = 1):
    n = len(probs)
    
    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))
      
    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

    filled_bins = np.unique(idx, axis=0)

    s = 0
    for bin in filled_bins:
        i = np.where((idx == bin).all(axis=1))[0]
        s += (len(i)/n) * (
            np.abs(np.mean(probs[i], axis=0) - np.mean(y_true[i], axis=0))**power
        ).sum()        

    return s


def label_resampling(probs):
    c = probs.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    y = np.zeros_like(probs)
    y[range(len(probs)), choices] = 1
    return y


def score_sampling(probs, samples = 10000, ece_function = None):

    probs = np.array(probs)

    return np.array(
        [
            ece_function(probs, label_resampling(probs)) for sample in range(samples)
        ]
    )


def pECE(probs, y_true, samples = 10000, ece_function = full_ECE):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    return 1 - (
        percentileofscore(
            score_sampling(
                probs,
                samples=samples,
                ece_function=ece_function
            ),
            ece_function(probs, y_true)
        ) / 100
    )

    