# Method for unpickling probabilities/logits saved in process of evaluation.

import pickle
from sklearn.model_selection import train_test_split 

# Example of unpickle
FILE_PATH = 'probs_resnet110_c10.p'


# Open file with pickled variables
def unpickle_probs(file, verbose = 0):
    with open(file, 'rb') as f:  # Python 3: open(..., 'rb')
        y1, y2 = pickle.load(f)  # unpickle the content
        
    if isinstance(y1, tuple):
        y_probs_val, y_val = y1
        y_probs_test, y_test = y2
    else:
        y_probs_val, y_probs_test, y_val, y_test = train_test_split(y1, y2.reshape(-1,1), test_size = len(y2)-5000, random_state=15)   # Splits the data in the case of pretrained models
        
    if verbose:    
        print("y_probs_val:", y_probs_val.shape)  # (5000, 10); Validation set probabilities of predictions
        print("y_true_val:", y_val.shape)  # (5000, 1); Validation set true labels
        print("y_probs_test:", y_probs_test.shape)  # (10000, 10); Test set probabilities
        print("y_true_test:", y_test.shape)  # (10000, 1); Test set true labels
        
    return ((y_probs_val, y_val), (y_probs_test, y_test))
    
    
if __name__ == '__main__':
    
    (y_probs_val, y_val), (y_probs_test, y_test) = unpickle_probs(FILE_PATH, True)
    
    print(y_probs_val[:10])