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

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )

from dirichlet import FullDirichletCalibrator
from utility.unpickle_probs import unpickle_probs
from utility.evaluation import evaluate, pECE, classwise_ECE, ECE

# For main method
import pickle
import datetime
import numpy as np
import argparse

from calibration.cal_methods import Dirichlet_NN, softmax, LogisticCalibration
import keras.backend as K

    
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

        models.append(model)
        
    return models
    
    
def get_test_pECE(models, probs, true, samples=1000, ece_function = ECE):
    
    preds = []
        
    for mod in models:
        preds.append(mod.predict(probs))
        
    return pECE(probs=np.mean(preds, axis=0), y_true=true, samples=samples, ece_function=ece_function)

    

def tune_dir_pECE(path, f, l2, mu, k_folds = 5, random_state = 15, verbose = True, double_learning = True, 
                model_dir = "models_dump", loss_fn = "sparse_categorical_crossentropy", comp_l2 = True,
                ece_function = ECE, use_logits = False, samples = 1000):
    
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
          
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    total_t1 = time.time()
    
    
    name = "_".join(f.split("_")[1:-1])
    print(name)

    # Read in the data
    FILE_PATH = join(path, f)
    (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)

    # Convert into probabilities

    # Convert into probabilities
    if use_logits:
        input_val = logits_val
        input_test = logits_test
    else:
        input_val = softmax(logits_val)  # Softmax logits
        input_test = softmax(logits_test)

    #Loop over lambdas to test
    models = kf_model(input_val, y_val, Dirichlet_NN, {"l2":l2, "mu":mu, "patience":15, "loss":loss_fn, "double_fit":double_learning, "comp":comp_l2, "use_logits":use_logits}, 
                                k_folds=k_folds, random_state=random_state, verbose=verbose) 
    
    # TODO separate function for pickling models and results
    now = datetime.datetime.now()
    fname = "data_results_keras_NN_%s_l2=%5f_mu=%5f_%s_%s.p" % (name, l2, mu, loss_fn, now.strftime("%Y_%m_%d_%H_%M_%S"))

    model_weights = []
    for mod in models:
        model_weights.append(mod.model.get_weights())


    with open(join(model_dir, fname), "wb") as f:
        pickle.dump((model_weights, (name, l2, mu)), f)


    pECE_score = get_test_pECE(models, input_test, y_test, ece_function = ece_function, samples = samples)
    results = [name, pECE_score, l2, mu]


    for mod in models:  # Delete old models and close class
        #del mod.model
        del mod
    K.clear_session()
    
        
    total_t2 = time.time()
    print("Time taken:", (total_t2-total_t1))
        
    return results   
 
 
 
if __name__== "__main__":
    
    PATH = join('..', '..', 'logits')
    
    files_10 = ('probs_resnet_wide32_c10_logits.p', 'probs_densenet40_c10_logits.p',
            'probs_lenet5_c10_logits.p', 'probs_resnet110_SD_c10_logits.p',
           'probs_resnet110_c10_logits.p', 'probs_resnet152_SD_SVHN_logits.p',
           'logits_pretrained_c10_logits.p', 'logits_pretrained_mnist_logits.p',
           'logits_pretrained_svhn_logits.p')

    files_100 = ('probs_resnet_wide32_c100_logits.p', 'probs_densenet40_c100_logits.p',
                 'probs_lenet5_c100_logits.p', 'probs_resnet110_SD_c100_logits.p',
                 'probs_resnet110_c100_logits.p', 'logits_pretrained_c100_logits.p')
    files = files_10 + files_100

    
    f_scores = join("..", "notebooks", "all_scores_val_test_ens_10_27.p")  # This file is generated by notebook "Dirichlet – Final Results (Table 3 & 4 and Supp. Table 13-18 and Supp. Figure 11)"
     
    with open(f_scores, "rb") as f:
        df_res = pickle.load(f)

               
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_folds', '-kf', type=int, default=5)
    parser.add_argument('--random_state', '-r', type=int, default=15)
    parser.add_argument('--double', '-d', type=bool, default=True)
    parser.add_argument('--comp_l2', action='store_true')
    parser.add_argument('--use_logits', action='store_true')
    parser.add_argument('--use_scipy', action='store_true')
    parser.add_argument('--model_dir', '-m_dir', type=str, default='models_best_pECE')
    parser.add_argument('--loss_fn', '-l', type=str, default='sparse_categorical_crossentropy')
    parser.add_argument('--method', '-m', type=str, default='dir_l2')  # dir_l2, dir_l2_mu, dir_l2_off, dir_l2_mu_off, mat_scale_l2, mat_scale_l2_off, mat_scale_l2_mu, mat_scale_l2_mu_off
    parser.add_argument('--ece_function', '-ece_f', type=str, default='ECE') 
    args = parser.parse_args()
    
    str_double = "_double" if args.double else ""
    str_model = "_scipy" if args.use_scipy else "_keras"
    str_logits = "_logits" if args.use_logits else ""
    
    print("Double learning:", args.double)
    print("Complementary L2:", args.comp_l2)
    
    all_results = []
    
    if args.ece_function == "ECE":
        ece_function = ECE
        str_ece = "guo"
        samples = 10000
    elif args.ece_function == "classwise_ECE":
        ece_function = classwise_ECE
        str_ece = "classwise"
        samples = 10000
    else:
        raise "Undefined ECE function!"
        
    print(ece_function)    
    
    for df in df_res:
    
        method = df.Method.iloc[0]

        if args.method != method:
            print("Pass method:", method)
            next

        else:    
            if "off" in method or "vec_scale" in method:
                comp_l2 = True
                str_comp = "_comp_l2"
            else:
                comp_l2 = False

            print("pECE for method:", method, "; comp_l2:", comp_l2)

            for i, row in df.iterrows():

                file = [f for f in files if row.Name in f][0]
                results = tune_dir_pECE(PATH, file, l2=row.L2, mu=row.mu, verbose=False, k_folds=args.k_folds, random_state=args.random_state, 
                                        double_learning = args.double, model_dir = "models_best_%s%s%s" % (method, str_comp, str_logits), loss_fn=args.loss_fn, 
                                        comp_l2 = comp_l2, use_logits=args.use_logits, ece_function = ece_function, samples = samples)   

                all_results.append(results + [method])
                
                    
    df_columns=["Name", "pECE", "L2", "mu", "Method"]
    df_res = pd.DataFrame(all_results, columns=df_columns)
    
    now = datetime.datetime.now()
    fname = "df_pECE_%s_%s_samples_%i_date_%s.p" % (str_ece, args.method, samples, now.strftime("%Y_%m_%d_%H_%M_%S"))
    
    with open(fname, "wb") as f:
        pickle.dump(df_res, f)