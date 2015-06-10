"""
"""
import os
import sys
import numpy as np
from ensembler import Ensembler
from classifier import logloss_mc
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV, _CalibratedClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plt

sys.path.append('E:\Competitions\OttoGroup\py_ml_utils\lib')
#sys.path.append('/home/sandrovegapons/anaconda/src/xgboost/wrapper')
import xgboost as xgb 

#from clf_xgboost import Clf_xgboost, my_train_xgboost

import pdb




def func_mlogloss(w, Xs, y):
    """
    """
    w = np.abs(w)
    sol = np.zeros((Xs[0].shape[0], 9))
    for i in range(len(w)):
        sol += Xs[i] * w[i]
    ll = logloss_mc(y, sol)
    reg = np.sqrt(np.sum(w**2)) * 0.001
    return ll + reg
    
    
class EC(BaseEstimator):
    """
    """
    def __init__(self, n_preds):
        super(EC, self).__init__()
        self.n_preds = n_preds
        self.epsilon = 1e-15
        
    def fit(self, X, y):
        Xs = np.hsplit(X, self.n_preds)
        x0 = np.ones(self.n_preds) / float(self.n_preds)        
        bounds = [(0,1)]*len(x0)   
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        res = minimize(func_mlogloss, x0, args=(Xs, y), method='SLSQP', 
                       bounds=bounds,
                       constraints=cons
                       )
#        print res.message, res.success
        self.w = res.x
        return self
    
    def predict_proba(self, X):
        Xs = np.hsplit(X, self.n_preds)
        y_pred = np.zeros((X.shape[0], 9))
        for i in range(len(self.w)):
            y_pred += Xs[i] * self.w[i]
        y_pred = y_pred / y_pred.sum(axis=1).reshape(-1, 1)
        y_pred = np.minimum(1-self.epsilon, np.maximum(self.epsilon, y_pred))
        return y_pred


def func_mlogloss_2(w, Xs, y):
    """
    """
    nc=9
    w = np.abs(w)
    sol = np.zeros((Xs[0].shape[0], 9))
    for i in range(len(w)):
        sol[:,i%nc] += Xs[i/nc][:,i%nc] * w[i]
    ll = logloss_mc(y, sol)
    reg = np.sqrt(np.sum(w**2)) * 0.01
    return ll + reg


class EC_2(BaseEstimator):
    """
    """
    def __init__(self, n_preds, w0=0):
        super(EC_2, self).__init__()
        self.n_preds = n_preds
        self.w0 = w0
        self.epsilon = 1e-15
        
    def fit(self, X, y):
        Xs = np.hsplit(X, self.n_preds)
#        if self.w0 == 0:
#            x0 = np.ones(self.n_preds * 9) / float(self.n_preds * 9) 
#        else:
        x0 = self.w0
        bounds = [(0,1)]*len(x0)   
        cons = ({'type':'eq','fun':lambda w: 9-sum(w)})
        res = minimize(func_mlogloss_2, x0, args=(Xs, y), method='SLSQP', 
                       bounds=bounds,
                       constraints=cons
                       )
#        print res.message, res.success
        self.w = res.x
        return self
    
    def predict_proba(self, X):
        Xs = np.hsplit(X, self.n_preds)
        y_pred = np.zeros((X.shape[0], 9))
        for i in range(len(self.w)):
            y_pred[:,i%9] += Xs[i/9][:,i%9] * self.w[i]
        y_pred = y_pred / y_pred.sum(axis=1).reshape(-1, 1)
        y_pred = np.minimum(1-self.epsilon, np.maximum(self.epsilon, y_pred))
        return y_pred


def scoring_mlogloss(clf, X_test, y_test):
    y_pred = clf.predict_proba(X_test)
    return logloss_mc(y_test, y_pred)

     
from sklearn.linear_model import SGDClassifier
class EC_3(BaseEstimator):
    """
    """
    def __init__(self, n_preds, n_classes=9):
        super(EC_3, self).__init__()
        self.n_preds = n_preds
        self.n_classes = n_classes
        self.epsilon = 1e-15
        self.sgd = SGDClassifier(alpha=0.0001, loss='log', class_weight='auto',
                                 penalty='l1')

        
    def fit(self, X, y):
      
        self.sgd.fit(X, y)
        return self

    
    def predict_proba(self, X):
        y_pred = self.sgd.predict_proba(X)
        y_pred = y_pred / y_pred.sum(axis=1).reshape(-1, 1)
        y_pred = np.minimum(1-self.epsilon, np.maximum(self.epsilon, y_pred))
        return y_pred


def func_mlogloss_4(w, Xs, y):
    """
    """
    w = np.abs(w)
    sol = np.zeros((Xs[0].shape[0], 9))
    for i in range(len(w)/2):
        sol[:,0] += Xs[i][:,0] * w[i]
        sol[:,1] += Xs[i][:,1] * w[i+1]
        sol[:,2] += Xs[i][:,2] * w[i+1]
        sol[:,3] += Xs[i][:,3] * w[i+1]
        sol[:,4] += Xs[i][:,4] * w[i]
        sol[:,5] += Xs[i][:,5] * w[i]
        sol[:,6] += Xs[i][:,6] * w[i]
        sol[:,7] += Xs[i][:,7] * w[i]
        sol[:,8] += Xs[i][:,8] * w[i]
    return logloss_mc(y, sol)
    
class EC_4(BaseEstimator):
    """
    """
    def __init__(self, n_preds):
        super(EC_4, self).__init__()
        self.n_preds = n_preds
        self.epsilon = 1e-15
        
    def fit(self, X, y):
        Xs = np.hsplit(X, self.n_preds)
        x0 = np.ones(self.n_preds * 2) / float(self.n_preds *2)        
        bounds = [(0,1)]*len(x0)   
#        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        res = minimize(func_mlogloss_4, x0, args=(Xs, y), 
                       method='SLSQP', bounds=bounds
#                       constraints=cons
                       )
#        print res.message, res.success
        self.w = res.x
        return self
    
    def predict_proba(self, X):
        Xs = np.hsplit(X, self.n_preds)
        y_pred = np.zeros((X.shape[0], 9))
        for i in range(len(self.w)/2):
            y_pred[:,0] += Xs[i][:,0] * self.w[i]
            y_pred[:,1] += Xs[i][:,1] * self.w[i+1]
            y_pred[:,2] += Xs[i][:,2] * self.w[i+1]
            y_pred[:,3] += Xs[i][:,3] * self.w[i+1]
            y_pred[:,4] += Xs[i][:,4] * self.w[i]
            y_pred[:,5] += Xs[i][:,5] * self.w[i]
            y_pred[:,6] += Xs[i][:,6] * self.w[i]
            y_pred[:,7] += Xs[i][:,7] * self.w[i]
            y_pred[:,8] += Xs[i][:,8] * self.w[i]
        y_pred = y_pred / y_pred.sum(axis=1).reshape(-1, 1)
        y_pred = np.minimum(1-self.epsilon, np.maximum(self.epsilon, y_pred))
        return y_pred
        
        
        
class DumClf(BaseEstimator):
    def fit(self, X, y):
        return self    
    def predict_proba(self, X):
        return X
        
    

class Ens_log_reg(Ensembler):
    """
    Logistic regression based ensembler...
    """
       
    def internal_processing(self, X, y, X_test):
        """
        """  
#        d_train = np.loadtxt('./data/X_train')
#        d_valid = np.loadtxt('./data/X_valid')
#        Xs = np.hsplit(X, 5)
#        Xs_cal = []
#        for i in range(len(Xs)):
#            cc = CalibratedClassifierCV(base_estimator=DumClf(), 
#                                        method='isotonic', cv=5)
#
#            cc.fit(Xs[i], y)
#            Xs_cal.append(cc.predict_proba(Xs[i]))
#        XX = np.hstack(Xs_cal)
        
#        Xts_cal = []
#        Xts = np.hsplit(X_test, 5)
#        for i in range(len(Xts)):
#            cc = CalibratedClassifierCV(base_estimator=DumClf(),
#                                        method='isotonic', cv=5)
#            cc.fit(Xts[i], (np.random.rand(len(Xts[i]))*10).astype(np.int32))
#            Xts_cal.append(cc.predict_proba(Xts[i]))
#        XX_test = np.hstack(Xts_cal)   
#        
#        print 'estoy aqui...'
#        
#        ec = EC(n_preds=self.n_preds)
#        ec.fit(X, y)
#        ew0 = ec.w
##        y_ens = ec.predict_proba(X_test)
#        
#        #validation
#        yv = ec.predict_proba(X)
#        print 'Weights: %s' %(ec.w)
#        print 'Validation log-loss: %s' %(logloss_mc(y, yv))
#        
#        
##        
##        cc = CalibratedClassifierCV(base_estimator=EC(n_preds=self.n_preds), 
##                                    method='isotonic', cv=10)
##                                    
##        cc.fit(X, y)
##        y_cal = cc.predict_proba(X_test)
##        
##        y1_pred = (y_ens + y_cal)/2.
##        
#        pdb.set_trace()
##        
###        
####      
#        w20 = np.ones(X.shape[1])
#        for i in range(len(w20)):
#            w20[i] = w20[i] * (ew0[i/9])
#        ec2 = EC_2(n_preds=self.n_preds, w0=w20)
#        ec2.fit(X, y)
##        y2_ens = ec2.predict_proba(X_test)
#        
#        #validation
#        yv2 = ec2.predict_proba(X)
#        print 'Weights: %s' %(ec2.w)
#        print 'Validation log-loss: %s' %(logloss_mc(y, yv2))
#        
#        pdb.set_trace()
        
#        cc2 = CalibratedClassifierCV(base_estimator=EC_2(n_preds=self.n_preds,
#                                                         w0=w20), 
#                                    method='isotonic', cv=10)
#                                    
#        cc2.fit(X, y)
#        y2_cal = cc2.predict_proba(X_test)
#        
##        y2_pred = (y2_ens + y2_cal)/2.
#        y2_pred = (y2_ens*2 + y2_cal*3)/5.
###        
###        
##        y_pred = (y1_pred + y2_pred)/2.
#        y_pred = y2_pred
##        
#        ec3 = EC_3(n_preds=self.n_preds)
#        ec3.fit(X, y)
##        y3_ens = ec4.predict_proba(X_test)
##        
##        #validation
#        yv3 = ec3.predict_proba(X)
##        print 'Weights: %s' %(ec4.w)
#        print 'Validation log-loss: %s' %(logloss_mc(y, yv3))
##        
#        cc3 = CalibratedClassifierCV(base_estimator=EC_3(n_preds=self.n_preds), 
#                                    method='isotonic', cv=10)
#                                    
#        cc3.fit(X, y)
#        y3_cal = cc3.predict_proba(X)
#        print 'Validation log-loss: %s' %(logloss_mc(y, y3_cal))
##        
#        y3_pred = (y3_ens + y3_cal)/2.
#        
#        y_pred = (y1_pred + y2_pred + y3_pred)/3.
#
##        
#        pdb.set_trace()
        
        
##        cc = CalibratedClassifierCV(base_estimator=ec, method='sigmoid', cv=5)
##        cc.fit(X, y)
##        y_pred = cc.predict_proba(X_test)
##        sss = StratifiedShuffleSplit(y, 1, test_size=0.5) 
        sss = StratifiedKFold(y, n_folds=10)
        for train_id, valid_id in sss:
            X0, X1 = X[train_id], X[valid_id]
            y0, y1 = y[train_id], y[valid_id] 
#            d0, d1 = d_train[train_id], d_train[valid_id]
            ec = EC(n_preds = self.n_preds)
            ec.fit(X0, y0)
            ew0 = ec.w
#            print ec.w
            y0_pred = ec.predict_proba(X1)
            print 1,logloss_mc(y1, y0_pred)
            
#            pp2 = np.where(y0_pred+0.01>1,1,y0_pred+0.01)
#            pp3 = np.where(y0_pred+0.001>1,1,y0_pred+0.001)
#            pp4 = np.where(y0_pred-0.01<0,0,y0_pred-0.01)
            pp4 = np.where(y0_pred>0.5, y0_pred-0.0001, y0_pred+0.0001)
            pp5 = np.where(y0_pred>0.5, y0_pred+0.0001, y0_pred-0.0001)
#            print 2, logloss_mc(y1, pp2)
#            print 3, logloss_mc(y1, pp3)
            print 3, logloss_mc(y1, pp4)
            print 4, logloss_mc(y1, pp5)
            
            
            
##            
#            tt = CalibratedClassifierCV(base_estimator=EC(n_preds=self.n_preds), 
#                                        method='isotonic', cv=5)
#            tt.fit(X0, y0)
#            y0_tt = tt.predict_proba(X1)
#            
#            
#            print 2,logloss_mc(y1, y0_tt)
##            
#            ym0 = (y0_tt+y0_pred)/2.
###            
#            print 3,logloss_mc(y1, ym0)
#            
#            y4 = np.where(ym0<0.05,0,ym0)
#            print 4, logloss_mc(y1, y4)
#            
#            y5 = np.where(ym0<0.2, ym0/2, ym0)
#            print 5, logloss_mc(y1, y5)
#            
#            yy = np.zeros(y0_pred.shape)
#            for i,j in enumerate(y1):
#                yy[i,j] = 1.
#            
#            plt.plot(y0_pred.reshape(-1), 'r^-')
#            plt.plot(y0_tt.reshape(-1), 'g-')
#            plt.plot(ym0.reshape(-1), 'b*-')
#            
##            plt.plot(yy.reshape(-1), 'y-')
#            plt.show()
#            
#            pdb.set_trace() 
            
            
            
#            print 4, logloss_mc(y1, (2*y0_tt+y0_pred)/3.)
#            print 5, logloss_mc(y1, (y0_tt+2*y0_pred)/3.)
#            
##            jj = CalibratedClassifierCV(base_estimator= CalibratedClassifierCV(base_estimator=EC(n_preds=self.n_preds), 
##                                        method='isotonic', cv=5), method='isotonic',
##                                        cv=3)
##            jj.fit(X0, y0)
##            y0_jj = jj.predict_proba(X1) 
##            print 6,logloss_mc(y1, y0_jj)
##            print 7,logloss_mc(y1, (y0_pred+y0_jj)/2.)
#            
#            mn = np.min(y0_pred)
#            mx = np.max(y0_pred)
#            mms = MinMaxScaler(feature_range=(mn, mx))
#            ymm = mms.fit_transform(y0_tt)
#            print 6, logloss_mc(y1, ymm)
#            print 7, logloss_mc(y1, (y0_pred+ymm)/2.)
#            
#            
##
####            
#            print '-----'
            w20 = np.ones(X0.shape[1])
            for i in range(len(w20)):
                w20[i] = w20[i] * (ew0[i/9])
                
            ec2 = EC_2(n_preds=self.n_preds, w0=w20)
            ec2.fit(X0, y0)
#            print ec2.w
            y02_pred = ec2.predict_proba(X1)
##            
##            
            print 4, logloss_mc(y1, y02_pred)
            pp8 = np.where(y02_pred>0.5, y02_pred-0.0001, y02_pred+0.0001)
            pp9 = np.where(y02_pred>0.5, y02_pred+0.0001, y02_pred-0.0001)
#            print 2, logloss_mc(y1, pp2)
#            print 3, logloss_mc(y1, pp3)
            print 5, logloss_mc(y1, pp8)
            print 6, logloss_mc(y1, pp9)
            
            print '---\n'
##            
##            gg = CalibratedClassifierCV(base_estimator=EC_2(n_preds=self.n_preds, w0=w20), 
##                            method='isotonic', cv=5)
##            gg.fit(X0, y0)
##            y0_gg = gg.predict_proba(X1)
##            
##            print 5, logloss_mc(y1, y0_gg)
##            
##            ym1 = (y0_gg+y02_pred)/2.
##            print 6, logloss_mc(y1, ym1)
###            
##            print 7, logloss_mc(y1, (ym0 + ym1)/2.)
#            
#            print '----' 
#           
#            ec3.fit(X0, y0)
#            y03_pred = ec3.predict_proba(X1)
#            print 4, logloss_mc(y1, y03_pred)
#            
#            hh = CalibratedClassifierCV(base_estimator=EC_3(n_preds=self.n_preds), 
#                method='isotonic', cv=5)
#            hh.fit(X0, y0)
#            y0_hh = hh.predict_proba(X1)
#            
#            print 5, logloss_mc(y1, y0_hh)
##            
#            ym3 = (y0_hh+y03_pred)/2.
#            print 6, logloss_mc(y1, ym3)
##            
#            print 7, logloss_mc(y1, (ym0+ym3)/2.)
#            print '------ ' 
#            print ' ' 
##            ecc = EC(n_preds=5)
###            y_aux = np.array([np.random.randint(9) for i in range(len(XX0))], dtype=np.int32)
###            pdb.set_trace()
##            
##            Xs0 = np.hsplit(X0, 5)
#            Xs1 = np.hsplit(X1, 5)
#            X0_cal = []
#            X1_cal = []
#            for i in range(len(Xs0)):
#                cc = CalibratedClassifierCV(base_estimator=DumClf(), 
#                                            method='isotonic', cv=5)
#    
#                cc.fit(d[i], y0)
#                X0_cal.append(cc.predict_proba(Xs0[i]))
#                X1_cal.append(cc.predict_proba(Xs1[i]))
#            XX0 = np.hstack(X0_cal) 
#            XX1 = np.hstack(X1_cal)
#            ecc.fit(XX0, y0)
#            y01_pred = ecc.predict_proba(XX1)
#            print 4,logloss_mc(y1, y01_pred)
#            
#            gg = CalibratedClassifierCV(base_estimator=EC(n_preds=5), 
#                                        method='isotonic', cv=5)
#            gg.fit(XX0, y0)
#            y0_gg = tt.predict_proba(XX1)
#            
#            print 5,logloss_mc(y1, y0_gg)
#            
#            print 6,logloss_mc(y1, (y0_gg+y01_pred)/2.)
#            
#            
#            
#            print ' '
            
#        pdb.set_trace()        
#        
#        Xs = np.hsplit(X, self.n_preds)
#        
#        #evaluating individual predictions
#        for i in range(len(Xs)):
#            print 'Solution %s, logloss: %s' %(i, logloss_mc(y, Xs[i]))
#        print ' '
#        
#        x0 = np.ones(self.n_preds) / float(self.n_preds)
#        
#        bounds = [(0,1)]*len(x0)
#    
#        res = minimize(func_mlogloss, x0, args=(Xs, y), 
#                       method='L-BFGS-B', bounds=bounds)
#        w = res.x
#        print res.message, res.success
#        print w, np.sum(w)
#        
#        ypv = np.zeros((X.shape[0], 9))
#        for i in range(len(w)):
#            ypv += Xs[i] * w[i]
#        print 'valid log-loss: %s' %(logloss_mc(y, ypv))
#        
#        
#        
##        pdb.set_trace()
#    
#        
#        Xs_test = np.hsplit(X_test, self.n_preds)
#        y_pred = np.zeros((X_test.shape[0], 9))
#        for i in range(len(w)):
#            y_pred += Xs_test[i] * w[i]
#            
#        pdb.set_trace()
         
        return y_pred       

    
        
    


    