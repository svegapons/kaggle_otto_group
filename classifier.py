"""
"""

import numpy as np
import pandas as pd


def logloss_mc(y_true, y_prob, epsilon=1e-15):
    """ Multiclass logloss. Code from competition benchmark.
    """
    # normalize
    y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
    y_prob = np.maximum(epsilon, y_prob)
    y_prob = np.minimum(1 - epsilon, y_prob)
    # get probabilities
    y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
    ll = - np.mean(np.log(y))
    return ll

    

class Clf:
    """
    Base class for our classifiers...
    """
    def __init__(self):        
        self.prefix = ''
       
    def process(self, X_train, y_train, X_valid, y_valid, X_test, 
                validating=True, testing=True, file_name=None, verbose=1):
        """
        """
        if file_name == None:
            file_name = str(np.random.randint(1000, 100000))
            
        if validating:
            if verbose: 
                print 'Validating...'
            pred_valid = self.train_validate(X_train, y_train, X_valid, 
                                             y_valid) 
                                             
            ll = logloss_mc(y_valid, pred_valid)
            if verbose:                
                print '#####################'
                print 'Validation log-loss: %s' %(ll)
                
            np.savetxt('./validation/valid_'+self.prefix+'_'+file_name+'_'+str(np.round(ll,decimals=4))+'.csv', pred_valid)
            
        if testing:
            if verbose:
                print 'Working on test set...'
            X = np.vstack((X_train, X_valid))
            y = np.hstack((y_train, y_valid))
            pred_test = self.train_test(X, y, X_test)
           
            np.savetxt('./test/test_'+self.prefix+'_'+file_name+'.csv', pred_test)
                    
        
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        pass
    
    def train_test(self, X, y, X_test):
        pass
        
    


    