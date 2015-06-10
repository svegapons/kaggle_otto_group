"""
"""
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder

#sys.path.append('/home/sandrovegapons/anaconda/src/xgboost/wrapper')
sys.path.append('E:\Competitions\OttoGroup\py_ml_utils\lib')
from xgboost import DMatrix 

from clf_xgboost import Clf_xgboost, my_train_xgboost
import pdb



class Clf_xgboost_2_levels(Clf_xgboost):
    """
    Simple xgboost based classifier.
    """
    def __init__(self, num_classes=9):
        """
        """
        self.prefix = 'xgb_2lv'
        #Classifier 0
        self.param0 = {}    
        self.param0['objective'] = 'multi:softprob'
        # scale weight of positive examples
        self.param0['eta'] = 0.015
        self.param0['gamma'] = 0.5
        self.param0['min_child_weight'] = 3.5
        self.param0['max_delta_step'] = 0
        self.param0['subsample'] = 0.3
        self.param0['colsample_bytree'] = 0.3
        self.param0['max_depth'] = 19
        self.param0['silent'] = 1
        self.param0['nthread'] = 7
        self.param0['eval_metric'] = 'mlogloss'
        self.param0['num_class'] = 2    
        self.num_round0 = 1200
        
        self.w0 = [1., 1.03]
        self.rt0_eta=1.00055
        self.rt0_ssp=1.0007
        self.rt0_clb=1.0007
        self.rt0_dpt=0.998
        
        #Classifier 1
        self.param1 = {}    
        self.param1['objective'] = 'multi:softprob'
        self.param1['eta'] = 0.01
        self.param1['gamma'] = 0.7
        self.param1['min_child_weight'] = 4
        self.param1['subsample'] = 0.5
        self.param1['max_depth'] = 17
        self.param1['max_delta_step'] = 10
        self.param1['colsample_bytree'] = 0.5 
        self.param1['silent'] = 1
        self.param1['nthread'] = 7
        self.param1['eval_metric'] = 'mlogloss'
        self.param1['num_class'] = 3   
        self.num_round1 = 1800
        
        self.w1 = [1., 1.17, 1.23]
        self.rt1_eta=1.00009
        self.rt1_ssp=1.0003
        self.rt1_clb=1.0003
        self.rt1_dpt=0.9996
        
        #Classifier 2
        self.param2 = {}    
        self.param2['objective'] = 'multi:softprob'
        self.param2['eta'] = 0.015
        self.param2['gamma'] = 0.7
        self.param2['min_child_weight'] = 3
        self.param2['subsample'] = 0.5
        self.param2['max_depth'] = 13
        self.param2['max_delta_step'] = 6
        self.param2['colsample_bytree'] = 0.5
        self.param2['silent'] = 1
        self.param2['nthread'] = 7
        self.param2['eval_metric'] = 'mlogloss'
        self.param2['num_class'] = 6   
        self.num_round2 = 1100
        
        self.w2 = [ 1.2, 1.2, 1., 1.2, 1.05, 1.1]
        self.rt2_eta=1.00055
        self.rt2_ssp=1.00055
        self.rt2_clb=1.00055
        self.rt2_dpt=0.9998
        
        
        
        
    
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        #training
        le = LabelEncoder()
        id_123 = np.logical_or(np.logical_or(y_train==1, y_train==2), 
                               y_train==3)  
        y0_train = np.zeros(len(y_train), dtype=np.int32)
        y0_train[id_123] = 1
        X0_train = np.copy(X_train) 
        y0_train = le.fit_transform(y0_train).astype(np.int32)
    
        X1_train = X_train[id_123]
        y1_train = y_train[id_123]
        y1_train = le.fit_transform(y1_train).astype(np.int32)
    
        X2_train = X_train[np.logical_not(id_123)]
        y2_train = y_train[np.logical_not(id_123)]    
        y2_train = le.fit_transform(y2_train).astype(np.int32)    
        
        #Validation
        id_123_valid = np.logical_or(np.logical_or(y_valid==1, y_valid==2), 
                               y_valid==3)  
        y0_valid = np.zeros(len(y_valid), dtype=np.int32)
        y0_valid[id_123_valid] = 1
        X0_valid = np.copy(X_valid) 
        y0_valid = le.fit_transform(y0_valid).astype(np.int32)
    
        X1_valid = X_valid[id_123_valid]
        y1_valid = y_valid[id_123_valid]
        y1_valid = le.fit_transform(y1_valid).astype(np.int32)
    
        X2_valid = X_valid[np.logical_not(id_123_valid)]
        y2_valid = y_valid[np.logical_not(id_123_valid)]    
        y2_valid = le.fit_transform(y2_valid).astype(np.int32)

        xg_valid = DMatrix(X_valid)        
        
        #Classifier 0
        w0_train = np.zeros(len(y0_train))
        for i in range(len(w0_train)):
            w0_train[i] = self.w0[int(y0_train[i])]
            
        xg0_train = DMatrix(X0_train, label=y0_train, weight=w0_train)  
        xg0_valid = DMatrix(X0_valid, label=y0_valid)
        watchlist0 = [(xg0_train,'train'), (xg0_valid, 'validation')]
    
        bst0 = my_train_xgboost(self.param0, xg0_train, self.num_round0, 
                                watchlist0, rt_eta=self.rt0_eta, 
                                rt_ssp=self.rt0_ssp, rt_clb=self.rt0_clb, 
                                rt_dpt=self.rt0_dpt)    
        y0_pred = bst0.predict(xg_valid).reshape(y_valid.shape[0], 2)
        
#        pdb.set_trace()
        
        #Classifier 1   
        w1_train = np.zeros(len(y1_train))
        for i in range(len(w1_train)):
            w1_train[i] = self.w1[int(y1_train[i])]
    
        xg1_train = DMatrix(X1_train, label=y1_train, weight=w1_train)  
        xg1_valid = DMatrix(X1_valid, label=y1_valid)
        watchlist1 = [(xg1_train,'train'), (xg1_valid, 'validation')]
        bst1 = my_train_xgboost(self.param1, xg1_train, self.num_round1, 
                                watchlist1, rt_eta=self.rt1_eta, 
                                rt_ssp=self.rt1_ssp, rt_clb=self.rt1_clb,
                                rt_dpt=self.rt1_dpt)
        y1_pred = bst1.predict(xg_valid).reshape(y_valid.shape[0], 3)
        
        #Classifier 2
        w2_train = np.zeros(len(y2_train))
        for i in range(len(w2_train)):
            w2_train[i] = self.w2[int(y2_train[i])]
    
        xg2_train = DMatrix(X2_train, label=y2_train, weight=w2_train)  
        xg2_valid = DMatrix(X2_valid, label=y2_valid)
        watchlist2 = [(xg2_train,'train'), (xg2_valid, 'validation')]
        bst2 = my_train_xgboost(self.param2, xg2_train, self.num_round2, 
                                watchlist2, rt_eta=self.rt2_eta, 
                                rt_ssp=self.rt2_ssp, rt_clb=self.rt2_clb,
                                rt_dpt=self.rt2_dpt)
        y2_pred = bst2.predict(xg_valid).reshape(y_valid.shape[0], 6)
        
        y_pred = np.zeros((y0_pred.shape[0], 9))
        y_pred[:,0] = y0_pred[:,0]*y2_pred[:,0]
        y_pred[:,1] = y0_pred[:,1]*y1_pred[:,0]
        y_pred[:,2] = y0_pred[:,1]*y1_pred[:,1]
        y_pred[:,3] = y0_pred[:,1]*y1_pred[:,2]
        y_pred[:,4] = y0_pred[:,0]*y2_pred[:,1]
        y_pred[:,5] = y0_pred[:,0]*y2_pred[:,2]
        y_pred[:,6] = y0_pred[:,0]*y2_pred[:,3]
        y_pred[:,7] = y0_pred[:,0]*y2_pred[:,4]
        y_pred[:,8] = y0_pred[:,0]*y2_pred[:,5] 
        
        return y_pred
        
        
        
    def train_test(self, X, y, X_test):
        """
        """
        #training
        le = LabelEncoder()
        id_123 = np.logical_or(np.logical_or(y==1, y==2), 
                               y==3)  
        y0 = np.zeros(len(y), dtype=np.int32)
        y0[id_123] = 1
        X0 = np.copy(X) 
        y0 = le.fit_transform(y0).astype(np.int32)
    
        X1 = X[id_123]
        y1 = y[id_123]
        y1 = le.fit_transform(y1).astype(np.int32)
    
        X2 = X[np.logical_not(id_123)]
        y2 = y[np.logical_not(id_123)]    
        y2 = le.fit_transform(y2).astype(np.int32)    
        
        xg_test = DMatrix(X_test) 
        
        #Classifier 0
        w0_train = np.zeros(len(y0))
        for i in range(len(w0_train)):
            w0_train[i] = self.w0[int(y0[i])]
            
        xg0_train = DMatrix(X0, label=y0, weight=w0_train)      
        bst0 = my_train_xgboost(self.param0, xg0_train, self.num_round0, 
                                rt_eta=self.rt0_eta, 
                                rt_ssp=self.rt0_ssp, rt_clb=self.rt0_clb, 
                                rt_dpt=self.rt0_dpt)    
        y0_pred = bst0.predict(xg_test).reshape(X_test.shape[0], 2)
        
        #Classifier 1   
        w1_train = np.zeros(len(y1))
        for i in range(len(w1_train)):
            w1_train[i] = self.w1[int(y1[i])]
    
        xg1_train = DMatrix(X1, label=y1, weight=w1_train)  
        bst1 = my_train_xgboost(self.param1, xg1_train, self.num_round1, 
                                rt_eta=self.rt1_eta, 
                                rt_ssp=self.rt1_ssp, rt_clb=self.rt1_clb,
                                rt_dpt=self.rt1_dpt)
        y1_pred = bst1.predict(xg_test).reshape(X_test.shape[0], 3)
        
        #Classifier 2
        w2_train = np.zeros(len(y2))
        for i in range(len(w2_train)):
            w2_train[i] = self.w2[int(y2[i])]
    
        xg2_train = DMatrix(X2, label=y2, weight=w2_train)  
        bst2 = my_train_xgboost(self.param2, xg2_train, self.num_round2, 
                                rt_eta=self.rt2_eta, 
                                rt_ssp=self.rt2_ssp, rt_clb=self.rt2_clb,
                                rt_dpt=self.rt2_dpt)
        y2_pred = bst2.predict(xg_test).reshape(X_test.shape[0], 6)
        
        y_pred = np.zeros((y0_pred.shape[0], 9))
        y_pred[:,0] = y0_pred[:,0]*y2_pred[:,0]
        y_pred[:,1] = y0_pred[:,1]*y1_pred[:,0]
        y_pred[:,2] = y0_pred[:,1]*y1_pred[:,1]
        y_pred[:,3] = y0_pred[:,1]*y1_pred[:,2]
        y_pred[:,4] = y0_pred[:,0]*y2_pred[:,1]
        y_pred[:,5] = y0_pred[:,0]*y2_pred[:,2]
        y_pred[:,6] = y0_pred[:,0]*y2_pred[:,3]
        y_pred[:,7] = y0_pred[:,0]*y2_pred[:,4]
        y_pred[:,8] = y0_pred[:,0]*y2_pred[:,5] 
        
        return y_pred              
        