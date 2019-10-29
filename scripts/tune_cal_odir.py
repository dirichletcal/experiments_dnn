# Functions for Dirichlet parameter tuning main class for CIFAR-100

from os.path import join
import time

# For main method
import pickle
import datetime
import numpy as np
import argparse

from tune_dirichlet_nn_slim import tune_dir_nn
    
    
if __name__== "__main__":
    
    PATH = join('..', 'logits')
    PATH_tunings = join('..', 'tunings')

    
    files = ('probs_resnet_wide32_c10_logits.p', 'probs_densenet40_c10_logits.p',
                'probs_lenet5_c10_logits.p', 'probs_resnet110_SD_c10_logits.p',
                'probs_resnet110_c10_logits.p', 'probs_resnet152_SD_SVHN_logits.p',
                'logits_pretrained_c10_logits.p', 'logits_pretrained_c100_logits.p', 
                'logits_pretrained_svhn_logits.p', 'logits_pretrained_mnist_logits.p',
                'probs_resnet_wide32_c100_logits.p', 'probs_densenet40_c100_logits.p',
                'probs_lenet5_c100_logits.p', 'probs_resnet110_SD_c100_logits.p',
                'probs_resnet110_c100_logits.p')               
               
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_nr', '-i', type=int, default=0, help="Index of file in list \"files_100\" to tune the models separately.")
    parser.add_argument('--mu_nr', '-m', type=int, default=-1, help="Index of mu's, to tune models parallel fashion")
    parser.add_argument('--k_folds', '-kf', type=int, default=5, help="How many cross-validation folds is done for tuning.")
    parser.add_argument('--random_state', '-r', type=int, default=15, help="Random state of splitting and training")
    parser.add_argument('--double', '-d', action='store_true', help="Fit second time with lower learning rates, to get a little bit better results.")
    parser.add_argument('--comp_l2', action='store_true', help="Complementary L2 learning, in paper it is called ODIR. Only off-diagonal values and intercept are regulated.")
    parser.add_argument('--use_logits', action='store_true', help="Use logits as inputs, instead of softmaxed values. Doing so we get matrix scaling instead of dirichlet scaling.")
    parser.add_argument('--use_scipy', action='store_true', help="Use scipy method for optimizing, instead of our neural network implementation")
    parser.add_argument('--model_dir', '-m_dir', type=str, default='model_weights', help="Dictionary for dumping fitted models for later use.")
    parser.add_argument('--loss_fn', '-l', type=str, default='sparse_categorical_crossentropy', help="specify loss function")
    parser.add_argument('--no_mus', action='store_true', help="No intercept tuning is done separately.")

    args = parser.parse_args()
    
    # 10 classes start L2 parameters from -5.0, 100 classes startfrom -2.0   
    if "c100" in files[args.file_nr]:
        start_from = -2.0
    else:
        start_from = -5.0

    # Set regularisation parameters to check through
    lambdas = np.array([10**i for i in np.arange(start_from, 7)])
    lambdas = sorted(np.concatenate([lambdas, lambdas*0.25, lambdas*0.5]))         
    mus = np.array([10**i for i in np.arange(start_from, 7)])

    
    # Check if mu's used for tuning
    if args.no_mus: 
        mus = [None]
        
    # Use specific mu from all mu's (in order to tune calibration methods parallelly)
    if args.mu_nr != -1:
        mus=[mus[args.mu_nr]]
    
    print(files[args.file_nr])
    print("Lambdas:", len(lambdas))
    print("Mus:", str(mus))
    print("Double learning:", args.double)    
    print("Complementary L2:", args.comp_l2)
    print("Using logits for Dirichlet:", args.use_logits)
    print("Using Scipy model instead of Keras:", args.use_scipy)
    
    df_res = tune_dir_nn(PATH, [files[args.file_nr]], lambdas=lambdas, mus=mus, verbose=False, k_folds=args.k_folds, random_state=args.random_state, double_learning = args.double, model_dir = args.model_dir, 
                         loss_fn=args.loss_fn, comp_l2 = args.comp_l2, use_logits = args.use_logits, use_scipy = args.use_scipy)    
    now = datetime.datetime.now()
    str_double = "_double" if args.double else ""
    str_comp = "_comp_l2" if args.comp_l2 else ""
    str_model = "_scipy" if args.use_scipy else "_keras"
    str_logits = "_logits" if args.use_logits else ""

    fname = join(PATH_tunings, "df_lambdas_l2_mu%s%s%s_f%i_%s_v6%s_%s.p" % (str_model, str_double, str_logits, args.file_nr, args.loss_fn, str_comp, now.strftime("%Y_%m_%d_%H_%M_%S")))
    
    with open(fname, "wb") as f:
        pickle.dump(df_res, f)