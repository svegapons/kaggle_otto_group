"""
"""
import os
import numpy as np
from ensembler import Ensembler
from classifier import logloss_mc
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier, IsotonicRegression
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
import pdb




def func_mlogloss(w, Xs, y):
    """
    """
    sol = np.zeros((Xs[0].shape[0], 9))
    for i in range(len(w)):
        sol += Xs[i] * w[i]
    return logloss_mc(y, sol)
    
    
class EC(BaseEstimator):
    """
    """
    def __init__(self, n_preds):
        super(EC, self).__init__()
        self.n_preds = n_preds
        
    def fit(self, X, y):
        Xs = np.hsplit(X, self.n_preds)
        x0 = np.ones(self.n_preds) / float(self.n_preds)        
        bounds = [(0,1)]*len(x0)   
#        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        res = minimize(func_mlogloss, x0, args=(Xs, y), 
                       method='L-BFGS-B', bounds=bounds
#                       constraints=cons
                       )
#        print res.message, res.success
        self.w = res.x
        return self
    
    def predict_proba(self, X):
        Xs = np.hsplit(X, self.n_preds)
        y_pred = np.zeros((X.shape[0], 9))
        for i in range(len(self.w)):
            y_pred += Xs[i] * self.w[i]
        return y_pred


def func_mlogloss_2(w, Xs, y):
    """
    """
    nc=9
    sol = np.zeros((Xs[0].shape[0], 9))
    for i in range(len(w)):
        sol[:,i%nc] += Xs[i/nc][:,i%nc] * w[i]
    return logloss_mc(y, sol)


class EC_2(BaseEstimator):
    """
    """
    def __init__(self, n_preds):
        super(EC_2, self).__init__()
        self.n_preds = n_preds
        
    def fit(self, X, y):
        Xs = np.hsplit(X, self.n_preds)
        x0 = np.ones(self.n_preds * 9) / float(self.n_preds * 9)        
        bounds = [(0,1)]*len(x0)   
#        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        res = minimize(func_mlogloss_2, x0, args=(Xs, y), 
                       method='SLSQP', bounds=bounds
#                       constraints=cons
                       )
        self.w = res.x
        return self
    
    def predict_proba(self, X):
        Xs = np.hsplit(X, self.n_preds)
        y_pred = np.zeros((X.shape[0], 9))
        for i in range(len(self.w)):
            y_pred[:,i%9] += Xs[i/9][:,i%9] * self.w[i]
        return y_pred
        
        
class FileClf(BaseEstimator):
    def fit(self, X, y):
        return self    
    def predict_proba(self, X):
        return X
        

def calibrate(X, y, X_test):
    """
    """
    yy = np.zeros(X.shape)
    for i, j in enumerate(y):
        yy[i,j] = 1
        

    Xt_cal = np.zeros(X_test.shape)
   
    for i in range(9):
        epsilon = 1e-10
        ist = IsotonicRegression(y_min=epsilon, y_max=1-epsilon, 
                                 increasing='auto',
                                 out_of_bounds='clip')
        ist.fit(X[:,i], yy[:,i])
        Xt_cal[:,i] = ist.transform(X_test[:,i])
   
    return (Xt_cal + X_test)/2.
        
    

class Ens_opt_cal(Ensembler):
    """
    Logistic regression based ensembler...
    """
       
    def internal_processing(self, X, y, X_test):
        """
        """  
        Xs = np.hsplit(X, 5)
        Xts = np.hsplit(X_test, 5)
        Xts_cal = []
        
        for i in range(len(Xs)):           
            Xts_cal.append(calibrate(Xs[i], y, Xts[i]))
         
        XX_test = np.hstack(Xts_cal)   
        
        ec = EC(n_preds=5)
        ec.fit(X, y)
        y_ens = ec.predict_proba(XX_test)
#        y_pred = ec.predict_proba(X_test)
        
        #validation
        yv = ec.predict_proba(X)
        print 'Weights: %s' %(ec.w)
        print 'Validation log-loss: %s' %(logloss_mc(y, yv))
        
        cc = CalibratedClassifierCV(base_estimator=EC(n_preds=5), 
                                    method='isotonic', cv=10)
                                    
        cc.fit(X, y)
        y_cal = cc.predict_proba(XX_test)
        
        y_pred = (y_ens + y_cal)/2.
         
        return y_pred       

    
        
    


    