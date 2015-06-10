"""
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import pdb


def normalize(X_train, X_valid, X_test, normalizer='StandardScaler'):
    """
    Normalizing the training, validation and test sets.
    """
    if normalizer == 'StandardScaler':
        norm = StandardScaler()
    elif normalizer == 'MinMaxScaler':
        norm = MinMaxScaler()
    else:
        raise Exception('Normalizer not supported')
    
    X_train = np.log(1+X_train)
    X_valid = np.log(1+X_valid)
    X_test = np.log(1+X_test)
    
    norm.fit(np.vstack((X_train, X_valid, X_test)))
    
    X_train = norm.transform(X_train)
    X_valid = norm.transform(X_valid)
    X_test = norm.transform(X_test)     
    
    return X_train, X_valid, X_test
    
    
#def feature_engineering(train, test, degree=2, 
#                                   interaction_only=True):
#    """
#    """
#    pef = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
#                             include_bias=False)
#    X_train = pef.fit_transform(train)
#    X_test = pef.fit_transform(train)
#    
#    return X_train, X_test



    
    
    
    


    