"""
"""
import numpy as np

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, NINLayer
from lasagne.nonlinearities import LeakyRectify, softmax
from lasagne.init import HeNormal
from lasagne.updates import adagrad

from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, clone
from sklearn.cross_validation import StratifiedKFold

from copy import copy, deepcopy

from classifier import logloss_mc

from theano import shared

from clf_nolearn import Clf_nolearn, My_NeuralNet, EarlyStopping, AdjustVariable, float32

import pdb


class ClfCal(BaseEstimator):
    """
    """
    def __init__(self, nn):
        super(ClfCal, self).__init__()
        self.nn = nn
    def fit(self, X, y):
        self.nn.fit(X, y)
        return self
    def predict_proba(self, X):
        y_pred = self.nn.predict_proba(X)
        return y_pred
        
            
class Clf_nolearn_simple(Clf_nolearn):
    """
    Simple nolearn based classifier...
    """
    def __init__(self, num_features=93, num_classes=9):
        """
        """
        self.prefix = 'nl_simp'
        self.num_features = num_features
        self.num_classes = num_classes 
        
        self.layers = [('input', InputLayer),
                  ('dropouti', DropoutLayer),
                  ('dense0', DenseLayer),
                  ('dropout0', DropoutLayer),
                  ('dense1', DenseLayer),
                  ('dropout1', DropoutLayer),
                  ('dense2', DenseLayer),
                  ('dropout2', DropoutLayer),
                  ('dense3', DenseLayer),
                  ('dropout3', DropoutLayer),
                  ('output', DenseLayer)]
                  
        self.early_stopping = EarlyStopping(patience=25)
        
        self.nn = My_NeuralNet(layers=self.layers,                     
                     input_shape=(None, self.num_features),
                     dropouti_p=0.12,
                     
                     dense0_num_units=900,
                     dense0_W=HeNormal(),
                     dense0_nonlinearity = LeakyRectify(leakiness=0.002),
                     dropout0_p=0.35,
                     
                     dense1_num_units=600,
                     dense1_W=HeNormal(),
                     dense1_nonlinearity = LeakyRectify(leakiness=0.002),
                     dropout1_p=0.2,

                     dense2_num_units=400,
                     dense2_W=HeNormal(),
                     dense2_nonlinearity = LeakyRectify(leakiness=0.002),
                     dropout2_p=0.1,
                     
                     dense3_num_units=300,
                     dense3_W=HeNormal(),
                     dense3_nonlinearity = LeakyRectify(leakiness=0.002),
                     dropout3_p=0.1,
                                          
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,
                     
                     update=adagrad,
                     update_learning_rate=shared(float32(0.01)),

                     batch_iterator_train=BatchIterator(batch_size=512),
                     on_epoch_finished=[
                     AdjustVariable('update_learning_rate', start=0.01,
                                    stop=0.005),
#                     AdjustVariable('update_momentum', start=0.1, stop=0.0),
                     self.early_stopping,
                     ],
                     eval_size=0.,
                     verbose=1,
                     max_epochs=300                     
                     )
                     
        self.nn2 = deepcopy(self.nn)
                     
                     
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        self.nn.max_epochs = 300
        self.nn.verbose=1
        self.nn.fit(X_train, y_train, X_valid, y_valid)
#        XX = np.vstack((X_train, X_valid[:len(X_valid)/2]))
#        yy = np.hstack((y_train, y_valid[:len(y_valid)/2]))
#        XXv = X_valid[len(X_valid)/2:]
#        yyv = y_valid[len(y_valid)/2:]
##        self.nn.dropouti_p=0.25
#        self.nn.fit(XX, yy, XXv, yyv)
        
        self.nn2.fit(X_train, y_train, X_valid, y_valid)
        
#        self.nn.fit(X_train, y_train)
        yp0 = self.nn.predict_proba(X_valid)
        print 'Nolearn log-loss: %s'%(logloss_mc(y_valid, yp0))
        
#        self.nn.max_epochs = self.early_stopping.best_valid_epoch
#        print self.early_stopping.best_valid_epoch
#        self.nn.verbose=0
#        
#        clf = ClfCal(self.nn)
#        cc = CalibratedClassifierCV(base_estimator=clf, method='isotonic',
#                                    cv=StratifiedKFold(y_train, n_folds=3))
#        cc.fit(X_train, y_train)
#        yp1= cc.predict_proba(X_valid)
#        print 'Calibrated log-loss: %s' %(logloss_mc(y_valid, yp1))
#        y_pred = (yp0+yp1)/2.
#        print 'Mean log-loss: %s' %(logloss_mc(y_valid, y_pred))
#        
#        self.cal_clf = cc
        y_pred = yp0
        
#        pdb.set_trace()
        
        return y_pred
        
    def train_test(self, X, y, X_test):
        """
        """
        self.nn.max_epochs = self.early_stopping.best_valid_epoch
#        self.nn.verbose=1
#        self.nn.eval_size=0.
#        self.nn.on_epoch_finished=[
#                     AdjustVariable('update_learning_rate', start=0.1,
#                                    stop=0.001)
#                                    ]
        
        self.nn.fit(X, y)
        yp0 = self.nn.predict_proba(X_test)
#        self.cal_clf.fit(X, y)        
#        yp1 = self.cal_clf.predict_proba(X_test)
#        y_pred = (yp0 + yp1)/2.
        y_pred = yp0
        return y_pred              
        
        

        

        

        