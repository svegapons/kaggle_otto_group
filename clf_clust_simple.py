"""
"""
import sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

sys.path.append('E:\Competitions\OttoGroup\py_ml_utils\lib')
#sys.path.append('/home/sandrovegapons/anaconda/src/xgboost/wrapper')
from xgboost import DMatrix 

from clf_xgboost import Clf_xgboost, my_train_xgboost

from classifier import Clf, logloss_mc
import pdb



class Clf_clust_simple(Clf):
    """
    
    """    
    def __init__(self, num_classes=9):
        """
        """
        self.prefix = 'clust_simp'
        self.param = {}    
        self.param['objective'] = 'multi:softprob'
        self.param['eta'] = 0.015        
        self.param['gamma'] = 8
        self.param['min_child_weight'] = 8
        self.param['max_delta_step'] = 1
        self.param['subsample'] = 0.5
        self.param['max_depth'] = 19
        self.param['silent'] = 1
        self.param['nthread'] = 6
        self.param['colsample_bytree'] = 0.5
        self.param['eval_metric'] = 'mlogloss'
        self.param['num_class'] = num_classes    
        self.num_round = 1400
        
        self.w = np.array([ 1.236, 1., 1.179, 1.230, 1.230, 1.073, 1.229, 1.173, 1.211])
    
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        nc = 10
        X = []
        y = []
        clt = MiniBatchKMeans(n_clusters=nc, batch_size=100)
        for i in range(9):
            XX = X_train[y_train==i]
            yy = y_train[y_train==i]
            lbs = clt.fit_predict(XX)
            ids = lbs < 7
            X.append(XX[ids])
            y.append(yy[ids])
        X = np.vstack(X)
        y = np.hstack(y)
        print X.shape
        
        w_train = np.zeros(len(y))
        for i in range(len(w_train)):
            w_train[i] = self.w[int(y[i])]
        xg_train = DMatrix(X, label=y, weight=w_train)  
        xg_valid = DMatrix(X_valid, label=y_valid)    
        watchlist = [(xg_train,'train'), (xg_valid, 'validation')]
        bst = my_train_xgboost(self.param, xg_train, self.num_round, watchlist)
        y_pred = bst.predict(xg_valid).reshape(X_valid.shape[0], 9)
        return y_pred
        
        
        
    def train_test(self, X, y, X_test):
        """
        """
        rf = RandomForestClassifier(n_estimators=500,
                                    class_weight='auto', max_features=0.9)
                                    
        rf.fit(X, y)
        yp0 = rf.predict_proba(X_test)
        rf = RandomForestClassifier(n_estimators=500, 
                            class_weight='auto', max_features=0.9)
                            
        cc = CalibratedClassifierCV(base_estimator=rf, method='isotonic',
                                    cv=StratifiedKFold(y, 3))
        cc.fit(X, y)
        yp1 = cc.predict_proba(X_test)
        
        y_pred = (yp0 + yp1)/2.
        
        return y_pred               
        