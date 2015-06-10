"""
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
import os

#np.random.seed(123)

def prepare_train_valid_test(path_train='./data/train.csv', 
                             path_test='./data/test.csv'):
    """
    Load train and test sets. Split train in train(80%) + valid(20%). Save
    train, valid and test.
    """
    df = pd.read_csv(path_train)
    data = df.values.copy()
    np.random.shuffle(data)
    X = data[:,1:-1].astype(float)
    labels = data[:,-1]
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    
    sss = StratifiedShuffleSplit(y, 1, test_size=0.20)
    
    for train_id, valid_id in sss:
        X_train, X_valid = X[train_id], X[valid_id]
        y_train, y_valid = y[train_id], y[valid_id]          
        
    df = pd.read_csv(path_test)
    data = df.values
    X_test = data[:, 1:].astype(float)
    
    np.savetxt('./data/X_train.csv', X_train)
    np.savetxt('./data/y_train.csv', y_train)
    np.savetxt('./data/X_valid.csv', X_valid)
    np.savetxt('./data/y_valid.csv', y_valid)    
    np.savetxt('./data/X_test.csv', X_test)   
    

def load_train_valid_test(folder='./data'):
    """
    Loading training, validation and test data.
    """
    X_train = np.loadtxt(os.path.join(folder, 'X_train.csv'))
    y_train = np.loadtxt(os.path.join(folder, 'y_train.csv'))
    X_valid = np.loadtxt(os.path.join(folder, 'X_valid.csv'))
    y_valid = np.loadtxt(os.path.join(folder, 'y_valid.csv'))
    X_test = np.loadtxt(os.path.join(folder, 'X_test.csv'))

    return X_train, y_train, X_valid, y_valid, X_test
    
   
def make_submission(y_pred, path_sample='./data/sampleSubmission.csv', 
                    path_submission='./submission/sub.csv'):
    """
    Write the prediction into the submission file.
    """
    sample_df = pd.read_csv(path_sample)
    preds = pd.DataFrame(y_pred, index=sample_df.id.values, 
                         columns=sample_df.columns[1:])
    preds.to_csv(path_submission, index_label='id')    
    print 'Submission written to file: %s' %(path_submission)


