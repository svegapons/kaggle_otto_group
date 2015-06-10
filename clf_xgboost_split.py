"""
"""
import sys
import numpy as np

sys.path.append('E:\Competitions\OttoGroup\py_ml_utils\lib')
#sys.path.append('/home/sandrovegapons/anaconda/src/xgboost/wrapper')
from xgboost import DMatrix, train 

from clf_xgboost import Clf_xgboost, my_train_xgboost
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import pdb


class Clf_xgboost_split(Clf_xgboost):
    """
    Simple xgboost based classifier.
    """
    def __init__(self, num_classes=9):
        """
        """
        self.prefix = 'xgb_splt'
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
        self.num_round = 850
        
        self.w = [ 1.236, 1., 1.179, 1.230, 1.230, 1.073, 1.229, 1.173, 1.211]
        
    
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        sss = StratifiedShuffleSplit(y_train, 1, test_size=0.5)    
        for train_id, valid_id in sss:
            X0_train, X1_train = X_train[train_id], X_train[valid_id]
            y0_train, y1_train = y_train[train_id], y_train[valid_id]  
            
        #First half
       
        w0_train = np.zeros(len(y0_train))
        for i in range(len(w0_train)):
            w0_train[i] = self.w[int(y0_train[i])]
        xg0_train = DMatrix(X0_train, label=y0_train, weight=w0_train)  
        xg0_valid = DMatrix(X1_train, label=y1_train)   
        xgv_valid = DMatrix(X_valid, label=y_valid)
        watchlist = [(xg0_train,'train'), (xg0_valid, 'validation0')]
        
#        bst0 = train(self.param, xg0_train, self.num_round, watchlist)
        bst0 = my_train_xgboost(self.param, xg0_train, self.num_round, watchlist)
        y0_pred = bst0.predict(xg0_valid).reshape(X1_train.shape[0], 9)
        yv_pred = bst0.predict(xgv_valid).reshape(X_valid.shape[0], 9)
        
        #Calibrated RF
        rf = RandomForestClassifier(n_estimators=600, criterion='gini', 
                                    class_weight='auto', max_features='auto')
        cal = CalibratedClassifierCV(rf, method='isotonic', cv=3)        
        cal.fit(X0_train, y0_train)
        y0_cal = cal.predict_proba(X1_train)
        yv_cal = cal.predict_proba(X_valid)
        
        #Second half
        ss = StandardScaler()
        y0_pred = ss.fit_transform(y0_pred)
        yv_pred = ss.fit_transform(yv_pred)
        y0_cal = ss.fit_transform(y0_cal)
        yv_cal = ss.fit_transform(yv_cal)
        X1_train = np.hstack((X1_train, y0_pred, y0_cal))
        X_valid = np.hstack((X_valid, yv_pred, yv_cal))        
        w1_train = np.zeros(len(y1_train))
        
#        self.param['eta'] = 0.05
        self.num_round = 450

        for i in range(len(w1_train)):
            w1_train[i] = self.w[int(y1_train[i])]
        xg1_train = DMatrix(X1_train, label=y1_train, weight=w1_train)    
        xg_valid = DMatrix(X_valid, label=y_valid)
        watchlist = [(xg1_train,'train'), (xg_valid, 'validation')]
        
#        bst1 = train(self.param, xg1_train, self.num_round, watchlist)
        bst1 = my_train_xgboost(self.param, xg1_train, self.num_round, watchlist)
        y_pred = bst1.predict(xg_valid).reshape(X_valid.shape[0], 9)

#        pdb.set_trace()
        return y_pred
        
        
    def train_test(self, X, y, X_test):
        """
        """
        sss = StratifiedShuffleSplit(y, 1, test_size=0.5)    
        for train_id, valid_id in sss:
            X0, X1 = X[train_id], X[valid_id]
            y0, y1 = y[train_id], y[valid_id]  
            
        #First half
        
        w0 = np.zeros(len(y0))
        for i in range(len(w0)):
            w0[i] = self.w[int(y0[i])]
        xg0_train = DMatrix(X0, label=y0, weight=w0)  
        xg0_test = DMatrix(X1, label=y1)   
        xgt_test = DMatrix(X_test)
        bst0 = my_train_xgboost(self.param, xg0_train, self.num_round)
        y0_pred = bst0.predict(xg0_test).reshape(X1.shape[0], 9)
        yt_pred = bst0.predict(xgt_test).reshape(X_test.shape[0], 9)
        
        #Calibrated RF
        rf = RandomForestClassifier(n_estimators=600, criterion='gini', 
                class_weight='auto', max_features='auto')
        cal = CalibratedClassifierCV(rf, method='isotonic', cv=3)
        cal.fit(X0, y0)
        y0_cal = cal.predict_proba(X1)
        yt_cal = cal.predict_proba(X_test)
        
        #Second half
        ss = StandardScaler()
        y0_pred = ss.fit_transform(y0_pred)
        yt_pred = ss.fit_transform(yt_pred)
        y0_cal = ss.fit_transform(y0_cal)
        yt_cal = ss.fit_transform(yt_cal)
        X1 = np.hstack((X1, y0_pred, y0_cal))
        X_test = np.hstack((X_test, yt_pred, yt_cal))  
        w1 = np.zeros(len(y1))
        
#        self.param['eta'] = 0.01
        self.num_round = 450

        for i in range(len(w1)):
            w1[i] = self.w[int(y1[i])]
        xg1_train = DMatrix(X1, label=y1, weight=w1)    
        xg_test= DMatrix(X_test)
        bst1 = my_train_xgboost(self.param, xg1_train, self.num_round)
        y_pred = bst1.predict(xg_test).reshape(X_test.shape[0], 9)
        
        return y_pred






                    
        