# Functions for Dirichlet parameter tuning


import numpy as np
import pandas as pd


from sklearn.metrics import log_loss, brier_score_loss
from os.path import join
import sklearn.metrics as metrics
import time
from sklearn.model_selection import KFold
from os.path import join
import os
import gc

from dirichlet import FullDirichletCalibrator
from utility.unpickle_probs import unpickle_probs
from utility.evaluation import evaluate, evaluate_rip

# For main method
import pickle
import datetime
import numpy as np
import argparse

from calibration.cal_methods import Dirichlet_NN, softmax, LogisticCalibration
#import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend

from sys import getsizeof


from keras import backend as K
    
def kf_model(input_val, y_val, fn, fn_kwargs = {}, k_folds = 5, random_state = 15, verbose = False):

    """
    K-fold task, mean and std of results are calculated over K folds
    
    Params:    
        input_val: (np.array) 2-D array holding instances (features) of validation set.
        y_val: (np.array) 1-D array holding y-values for validation set.
        fn: (class) a method used for calibration
        l2: (float) L2 regulariation value.
        k_folds: (int) how many crossvalidation folds are used.
        comp_l2: (bool) use reversed L2 matrix for regulariation (default = False)
    
    returns: 
        mean_error, mean_ece, mean_mce, mean_loss, mean_brier, std_loss, std_brier
    """
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    kf_results = []
    models = []

    for train_index, test_index in kf.split(input_val):
        X_train_c, X_val_c = input_val[train_index], input_val[test_index]
        y_train_c, y_val_c = y_val[train_index], y_val[test_index]
        
        t1 = time.time()

        model = fn(**fn_kwargs)
        model.fit(X_train_c, y_train_c)
        print("Model trained:", time.time()-t1)


        probs_holdout = model.predict_proba(X_val_c)
        error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate_rip(probs_holdout, y_val_c, verbose=False)
        kf_results.append([error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])
        models.append(model)
        
    return (models, ((np.mean(kf_results, axis = 0)), (np.std(np.array(kf_results)[:, -2:], axis = 0))))
    
    
def one_model(input_val, y_val, fn, fn_kwargs = {}, k_folds = 1, random_state = 15, verbose = False):

    """
    1-fold task, mean and std of results are calculated over 1 folds
    
    Params:    
        input_val: (np.array) 2-D array holding instances (features) of validation set.
        y_val: (np.array) 1-D array holding y-values for validation set.
        fn: (class) a method used for calibration
        l2: (float) L2 regulariation value.
        k_folds: (int) how many crossvalidation folds are used.
        comp_l2: (bool) use reversed L2 matrix for regulariation (default = False)
    
    returns: 
        mean_error, mean_ece, mean_mce, mean_loss, mean_brier, std_loss, std_brier
    """
    
    kf_results = []
    models = []
    
    t1 = time.time()

    model = fn(**fn_kwargs)
    model.fit(input_val, y_val)
    
    print("Model trained:", time.time()-t1)

    probs_holdout = model.predict_proba(input_val)
    error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate_rip(probs_holdout, y_val, verbose=False)
    kf_results.append([error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])
    models.append(model)
        
    return (models, ((np.mean(kf_results, axis = 0)), (np.std(np.array(kf_results)[:, -2:], axis = 0))))
    
    
def get_test_scores(models, probs, true):
    
    scores = []
        
    for mod in models:
        preds = mod.predict(probs)
        scores.append(evaluate_rip(probs=preds, y_true=true, verbose=False))
        
    return np.mean(scores, axis=0)
    
def get_test_scores2(models, probs, true):
    
    preds = []
        
    for mod in models:
        preds.append(mod.predict(probs))
        
    return evaluate_rip(np.mean(preds, axis=0), y_true=true, verbose=False)
    

def tune_dir_nn(path, files, lambdas, mus, k_folds = 5, random_state = 15, verbose = True, double_learning = False, model_dir = "models_dump", loss_fn = "sparse_categorical_crossentropy", 
                comp_l2 = True, use_logits = False, use_scipy = False):
    
    """
    
    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict", 
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        comp_l2 (bool): use reversed L2 matrix for regulariation (default = False)
        
    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.
    
    """
    
    df_columns=["Name", "L2", "mu", "Error", "ECE", "ECE2", "ECE_CW", "ECE_CW2", "ECE_FULL", "ECE_FULL2", "MCE", "MCE2", "Loss", "Brier", "Loss_std", "Brier_std", 
               "Error_test", "ECE_test", "ECE2_test", "ECE_CW_test", "ECE_CW2_test", "ECE_FULL_test", "ECE_FULL2_test", "MCE_test", "MCE2_test", "Loss_test", "Brier_test"]
    
    results = []
    results2 = []
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    total_t1 = time.time()
    
    for i, f in enumerate(files):
        
        name = "_".join(f.split("_")[1:-1])
        print(name)
        t1 = time.time()

        # Read in the data
        FILE_PATH = join(path, f)
        (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)
              
        # Convert into probabilities
        if use_logits:
            input_val = logits_val
            input_test = logits_test
        else:
            input_val = softmax(logits_val)  # Softmax logits
            input_test = softmax(logits_test)
        
        error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = evaluate_rip(softmax(logits_test), y_test, verbose=False)  # Uncalibrated results
        print("Uncal: Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f; brier %f" % (error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))        
        
        #Loop over lambdas to test
        for l2 in lambdas:            
            for mu in mus:                
            #Cross-validation
            
                if mu is None:
                    mu = l2
            
                if use_scipy:
                    temp_res = kf_model(input_val, y_val, LogisticCalibration, {"C":np.true_divide(1, l2), }, k_folds=k_folds, random_state=random_state, verbose=verbose)
                else:
                    if k_folds > 1:
                        temp_res = kf_model(input_val, y_val, Dirichlet_NN, {"l2":l2, "mu":mu, "patience":15, "loss":loss_fn, "double_fit":double_learning, "comp":comp_l2, "use_logits":use_logits}, 
                                        k_folds=k_folds, random_state=random_state, verbose=verbose)    
                    else:
                        temp_res = one_model(input_val, y_val, Dirichlet_NN, {"l2":l2, "mu":mu, "patience":15, "loss":loss_fn, "double_fit":double_learning, "comp":comp_l2, "use_logits":use_logits}, 
                                        k_folds=k_folds, random_state=random_state, verbose=verbose)  
                                    
                (models, ((avg_error, avg_ece, avg_ece2, avg_ece_cw, avg_ece_cw2, avg_ece_full, avg_ece_full2, avg_mce, avg_mce2, avg_loss, avg_brier), (std_loss, std_brier))) = temp_res  

                # TODO separate function for pickling models and results
                now = datetime.datetime.now()
                fname = "data_results_keras_NN_%s_l2=%5f_mu=%5f_%s_%s.p" % (name, l2, mu, loss_fn, now.strftime("%Y_%m_%d_%H_%M_%S"))
                    
                model_weights = []
                for mod in models:
                    if not use_scipy:
                        model_weights.append(mod.model.get_weights())
                    else:
                        model_weights.append([mod.coef_, mod.intercept_])
                    
                with open(join(model_dir, fname), "wb") as f:
                        pickle.dump((model_weights, temp_res[1], (name, l2, mu)), f)

                print("L2 = %f, Mu= %f, Validation Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f; brier %f" % (l2, mu, avg_error, avg_ece, avg_ece2, avg_ece_cw, avg_ece_cw2, avg_ece_full, avg_ece_full2, avg_mce, avg_mce2, avg_loss, avg_brier)) 
                
                error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = get_test_scores(models, input_test, y_test)
                print("L2 = %f, Mu= %f, Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f; brier %f" % (l2, mu, error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier))                             
                results.append([name, l2, mu, avg_error, avg_ece, avg_ece2, avg_ece_cw, avg_ece_cw2, avg_ece_full, avg_ece_full2, avg_mce, avg_mce2, avg_loss, avg_brier, std_loss, std_brier, error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])
                
                print("Ensambled results:")
                error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier = get_test_scores2(models, input_test, y_test)
                print("L2 = %f, Mu= %f, Test Error %f; ece %f; ece2 %f; ece_cw %f; ece_cw2 %f; ece_full %f; ece_full2 %f; mce %f; mce2 %f; loss %f; brier %f" % (l2, mu, error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier)) 
                results2.append([name, l2, mu, avg_error, avg_ece, avg_ece2, avg_ece_cw, avg_ece_cw2, avg_ece_full, avg_ece_full2, avg_mce, avg_mce2, avg_loss, avg_brier, std_loss, std_brier, error, ece, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, loss, brier])
                                
                
                # Garbage collection, I had some issues with newer version of Keras.
                K.clear_session()
                for mod in models:  # Delete old models and close class
                    del mod
                del models                
                del temp_res
                K.clear_session()                    
                gc.collect()

        
        t2 = time.time()
        print("Time taken:", (t2-t1), "\n")
        
    total_t2 = time.time()
    print("Total time taken:", (total_t2-total_t1))
    
    df = pd.DataFrame(results, columns=df_columns)
    df2 = pd.DataFrame(results2, columns=df_columns)

        
    return (df, df2)