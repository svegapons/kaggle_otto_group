"""
"""
import sys
import numpy as np

sys.path.append('E:\Competitions\OttoGroup\py_ml_utils\lib')
#sys.path.append('/home/sandrovegapons/anaconda/src/xgboost/wrapper')
from xgboost import DMatrix 
import pdb

from clf_xgboost import Clf_xgboost, my_train_xgboost

if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,



class Clf_xgboost_simple(Clf_xgboost):
    """
    Simple xgboost based classifier.
    """
    def __init__(self, num_classes=9):
        """
        """
        self.prefix = 'xgb_simp'
        self.param = {}    
        self.param['objective'] = 'multi:softprob'
        self.param['eta'] = 0.02        
        self.param['gamma'] = 0.8
        self.param['min_child_weight'] = 4
        #    param['max_delta_step'] = 5
        self.param['subsample'] = 0.5
        self.param['max_depth'] = 15
        self.param['silent'] = 1
        self.param['nthread'] = 6
        self.param['colsample_bytree'] = 0.5
        self.param['eval_metric'] = 'mlogloss'
        self.param['num_class'] = num_classes    
        self.num_round = 10000
        
        
        self.w = [ 1.236, 1., 1.179, 1.230, 1.230, 1.073, 1.229, 1.173, 1.211]
    
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        self.feats = np.ones(X_train.shape[1], dtype=np.bool)
        rd = np.random.randint(0, X_train.shape[1], 3)
        self.feats[rd] = False
        X_train = X_train[:, self.feats]
        X_valid = X_valid[:, self.feats]
        
        w_train = np.zeros(len(y_train))
        for i in range(len(w_train)):
            w_train[i] = self.w[int(y_train[i])]
        xg_train = DMatrix(X_train, label=y_train, weight=w_train)  
        xg_valid = DMatrix(X_valid, label=y_valid)    
        watchlist = [(xg_train,'train'), (xg_valid, 'validation')]
        self.seed = np.random.randint(0,10000)
        bst = my_train_xgboost(self.param, xg_train, self.num_round, watchlist,
                               early_stopping_rounds=100, seed=self.seed)
        self.best_n_iters = bst.best_iteration
        print self.best_n_iters
        y_pred = bst.predict(xg_valid,  ntree_limit=self.best_n_iters)
#        pdb.set_trace()
        return y_pred
        
        
    def train_test(self, X, y, X_test):
        """
        """
        X = X[:, self.feats]
        X_test = X_test[:, self.feats]

        w_train = np.zeros(len(y))
        for i in range(len(w_train)):
            w_train[i] = self.w[int(y[i])]
        xg_train = DMatrix(X, label=y, weight=w_train)  
        xg_test = DMatrix(X_test)   
        self.num_round = self.best_n_iters
        bst = my_train_xgboost(self.param, xg_train, self.num_round, 
                               seed=self.seed)
        y_pred = bst.predict(xg_test).reshape(X_test.shape[0], 9)
        return y_pred               
        