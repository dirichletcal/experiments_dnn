### Guo's Vector Scaling optimizing with neural network


from os.path import join
from calibration.cal_methods import cal_results, VectorScaling_NN, TemperatureScaling
import pickle
import datetime
import os
import argparse


PATH = join('..', 'logits')
PATH_tunings = join('..', 'tunings')

if not os.path.exists(PATH_tunings):
    os.makedirs(PATH_tunings)

files_10 = ('probs_resnet_wide32_c10_logits.p', 'probs_densenet40_c10_logits.p',
            'probs_lenet5_c10_logits.p', 'probs_resnet110_SD_c10_logits.p',
           'probs_resnet110_c10_logits.p', 'probs_resnet152_SD_SVHN_logits.p',
           'logits_pretrained_c10_logits.p', 'logits_pretrained_mnist_logits.p',
           'logits_pretrained_svhn_logits.p')

files_100 = ('probs_resnet_wide32_c100_logits.p', 'probs_densenet40_c100_logits.p',
             'probs_lenet5_c100_logits.p', 'probs_resnet110_SD_c100_logits.p',
             'probs_resnet110_c100_logits.p', 'logits_pretrained_c100_logits.p')

files = files_10 + files_100

calibrators = ["TemperatureScaling", "VectorScaling"]

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--cal_method', '-c', type=str, default="VectorScaling", help="Select one of the following calibration methods: %s." % calibrators)
    
    args = parser.parse_args()


    if args.cal_method == "VectorScaling":
        method = VectorScaling_NN
        str_cal = "vec_scale"
    elif args.cal_method == "TemperatureScaling":
        method = TemperatureScaling
        str_cal = "temp_scale"
    else:
        print("Select calibrator from %s" % calibrators)
        

    df_guo = cal_results(method, PATH, files, approach = "all", input="logits")

    now = datetime.datetime.now()
    fname = join(PATH_tunings, "df_guo_%s_%s.p" % (str_cal, now.strftime("%Y_%m_%d_%H_%M_%S")))  # Save straight into tunings folder

    with open(fname, "wb") as f:
        pickle.dump(df_guo, f)