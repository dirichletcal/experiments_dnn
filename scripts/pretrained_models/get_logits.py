import torch
import pickle
from torch.autograd import Variable
from utee import selector
import argparse
import numpy as np

def get_logits(data = "cifar10", output = "c10"):

    model_raw, ds_fetcher, is_imagenet = selector.select(data, cuda=False)
    ds_val = ds_fetcher(batch_size=100, train=False, val=True)

    outputs = []
    targets = []
    model_raw.eval()
    
    print("Start generating logits")

    for idx, (data, target) in enumerate(ds_val):
        data =  Variable(torch.FloatTensor(data)) #.cuda()
        outputs.append(model_raw(data).detach().numpy())
        targets.append(target.detach().numpy())

        
        if idx % 10 == 0:
            print("Iteration:", idx)
            
    print("Saving logits")
            
    with open("logits_%s_pretrained.p" % output, "wb") as f:
        pickle.dump((np.concatenate(outputs, axis=0), np.concatenate(targets, axis=0)), f)
        
    print("Done")

if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default="cifar10")
    parser.add_argument('--out', '-o', type=str, default="c10")

    args = parser.parse_args()
    
    
    get_logits(args.data, args.out)