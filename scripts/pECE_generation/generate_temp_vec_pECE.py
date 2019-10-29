import numpy as np
import pandas as pd
from os.path import join

# Imports to get "utility" package
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )

from calibration.cal_methods import TemperatureScaling, VectorScaling_NN
from dirichlet import FullDirichletCalibrator
import pickle
from utility.unpickle_probs import unpickle_probs
from utility.evaluation import pECE, classwise_ECE, score_sampling, ECE
from time import time
import datetime
import argparse


def temp_vec_pECE(path_files, files, ece_function = ECE, samples = 10000):

    dict_res_temp = {}
    dict_res_vec = {}

    for file in files:
        FILE_PATH = join(path_files, file)
        (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)

        name = "_".join(file.split("_")[1:-1])  # file_name

        print(name)

        # TemperatureScaling
        temp = TemperatureScaling()
        temp.fit(logits_val, y_val)
        preds_tempscale = temp.predict(logits_test)
        
        vec = VectorScaling_NN()
        vec.fit(logits_val, y_val, verbose=False)
        preds_vec = vec.predict(logits_test)

        t1 = time()
        res1 = pECE(probs=preds_tempscale, y_true=y_test, samples=samples, ece_function = ece_function)
        dict_res_temp[name] = res1
        print("Temp_scale; Name", name, "pECE:", res1)
        
        res2 = pECE(probs=preds_vec, y_true=y_test, samples=samples, ece_function = ece_function)
        dict_res_vec[name] = res2
        print("Vec_scale; Name", name, "pECE:", res2)
        
        print("Time spent:", time() - t1)
        
    return (dict_res_temp, dict_res_vec)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ece_function', '-ece_f', type=str, default='ECE', help="Choose between ECE and classwise_ECE.") 
    args = parser.parse_args()
    
    if args.ece_function == "ECE":
        ece_function = ECE
        str_ece = "confidence"
        samples = 10000
    elif args.ece_function == "classwise_ECE":
        ece_function = classwise_ECE
        str_ece = "classwise"
        samples = 10000
    else:
        raise "Undefined ECE function!"
    
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
    
    res = temp_vec_pECE(PATH, files, ece_function = ece_function, samples = samples)
    
    df1 = pd.DataFrame(list(res[0].items()), columns=["Name", "pECE"]).assign(Method = "temp_scale")
    df2 = pd.DataFrame(list(res[1].items()), columns=["Name", "pECE"]).assign(Method = "vec_scale")
    df_res = pd.concat([df1, df2]).reset_index()

    print(df_res)
    
    now = datetime.datetime.now()
    fname = "df_temp_vec_pECE_%s_%s.p" % (str_ece, now.strftime("%Y_%m_%d_%H_%M_%S"))
    
    with open(fname, "wb") as f:
        pickle.dump(df_res, f)