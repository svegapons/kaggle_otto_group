"""
"""
import os
import numpy as np

from load_save import load_train_valid_test, make_submission, prepare_train_valid_test
from preprocessing import normalize
from clf_nolearn_simple import Clf_nolearn_simple
from clf_nolearn_2_levels import Clf_nolearn_2_levels
from clf_xgboost_simple import Clf_xgboost_simple
from clf_xgboost_2_levels import Clf_xgboost_2_levels
from clf_xgboost_split import Clf_xgboost_split
from clf_rf_simple import Clf_rf_simple
from clf_nolearn_simple_play import Clf_nolearn_simple_play
from clf_clust_simple import Clf_clust_simple
from ens_log_reg import Ens_log_reg
from ens_opt_cal import Ens_opt_cal
import pdb

def main(data_folder='./data', path_submission='sub.csv'):
    """
    """
#    pdb.set_trace()
#    Load train, test and sample submission
#    print 'Loading data...'
#    X_train, y_train, X_valid, y_valid, X_test = load_train_valid_test(folder=data_folder)  
#    y_train = y_train.astype(np.int32)
#    y_valid = y_valid.astype(np.int32)
#    
#    #Data normalization
#    print 'Normalizing data...'
#    X_train, X_valid, X_test = normalize(X_train, X_valid, X_test, 
#                                             normalizer='StandardScaler')
#    #Feature engineering
#    print 'Feature engineering...'
##    X_train, X_test = feat_eng(X_train, X_test)
##    
##    #Applying classifiers...                               
#    print 'Classifying...'
#    num_features = X_train.shape[1]
#    num_classes = len(np.unique(y_train))
    
#    clf = Clf_nolearn_simple(num_features, num_classes)
#    clf = Clf_nolearn_2_levels(num_features, num_classes)
#    clf = Clf_nolearn_simple_play(num_features, num_classes)
#    clf = Clf_xgboost_simple(num_classes)
#    clf = Clf_xgboost_2_levels(num_classes)
#    clf = Clf_xgboost_split(num_classes)
#    clf = Clf_rf_simple()
#    clf = Clf_clust_simple()
###    
#    clf.process(X_train, y_train, X_valid, y_valid, X_test, 
#                validating=True, testing=True, file_name=None, verbose=1)
####  
#    print 'Ensembling...'              
    ens = Ens_log_reg()
####    ens = Ens_opt_cal()
    y_pred = ens.process()
######    
    make_submission(y_pred)
#    print 'Writing submission...'
#    make_submission(y_pred, encoder, sample_df, path_submission)


def split_train_valid():
    prepare_train_valid_test(path_train='./data/train.csv', path_test='./data/test.csv')
    
    


def modif_submission(path_sub='best_sub.csv'):
    """
    """
    arr = np.loadtxt(path_sub, skiprows=1, delimiter=',', usecols=range(1,10))
    pdb.set_trace()
    sub = np.where(arr>0.5, arr+0.0001, arr-0.0001)
    sub = np.where(sub<0,0,sub)
    sub = np.where(sub>1,1,sub)
    make_submission(sub)
        
    
    
    
    

    
if __name__=='__main__':
#    main()
#    main()
    modif_submission()
    