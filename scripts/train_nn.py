# Main method for training all the Neural Networks for Cifar-10, cifar-100 and so on

from train_resnet_wide import train_wide
from train_resnet_sd import train_svhn, train_sd
from train_resnet_densenet import train_dense
from train_resnet_cifar import train_resnet
from train_lenet_cifar import train_lenet

import argparse

available_datasets = ["CIFAR-10", "CIFAR-100", "SVHN"]  # note, in case of SVHN it is important to have data downloaded and put into correct location.
dict_models = {"resnet110":train_resnet,
                "resnet110_SD":train_sd,
                "densenet40":train_dense, 
                "resnet_wide32":train_wide,
                "lenet":train_lenet
            }

if __name__ == '__main__':

    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=13, help="Random seed for train and validation split")
    parser.add_argument('--model', '-m', type=str, default="lenet", 
                        help="Select one of the available models: %s" % str(dict_models.keys()))
    parser.add_argument('--data', '-d', type=str, default="CIFAR-10", 
                        help="Select one dataset: %s, note that SVHN works only with resnet110_SD" % str(available_datasets))

    args = parser.parse_args()
    seed = args.seed
    model_str = args.model
    data_str = args.data
    
    if model_str not in dict_models.keys():
        print("Please select one of the following datasets:", dict_models.keys())
    
    if data_str not in available_datasets:
        print("Please select one of the following datasets:", available_datasets)
        
    if data_str == "SVHN":
        if model_str != "resnet110_SD":
            print("Dataset SVHN is only suitable for model resnet110_SD")
        else:
            num_classes = 10
            method = train_svhn
            
    elif data_str == "CIFAR-10":
    
        num_classes = 10
        method = dict_models[model_str]
        
    else:  # CIFAR-100
        num_classes = 100
        method = dict_models[model_str]
        
        
    print("Start model training process for model %s on dataset %s\n\n" % (model_str, data_str)) 
    method.train(seed = seed, num_classes = num_classes)
    
    print("Start model logits generation process for model %s on dataset %s\n\n" % (model_str, data_str)) 
    method.gen_logits(seed = seed, num_classes = num_classes)
    