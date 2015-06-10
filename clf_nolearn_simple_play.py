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
from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from copy import deepcopy

from classifier import logloss_mc

from theano import shared

from clf_nolearn import Clf_nolearn, My_NeuralNet, EarlyStopping, AdjustVariable, float32, OneOneStopping

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
        
            
class Clf_nolearn_simple_play(Clf_nolearn):
    """
    Simple nolearn based classifier...
    """
    def __init__(self, num_features=93, num_classes=9):
        """
        """
        self.prefix = 'nl_simp_play'
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
                  
#        self.early_stopping = EarlyStopping(patience=25)
        self.oneone_stopping = OneOneStopping(ratio=0.9)
        self.oneone_stopping2 = OneOneStopping(ratio=0.9)
        self.oneone_stopping3 = OneOneStopping(ratio=0.85)
        
        self.nn = My_NeuralNet(layers=self.layers,                     
                     input_shape=(None, self.num_features),
                     dropouti_p=0.1,
                     
                     dense0_num_units=900,
                     dense0_W=HeNormal(),
                     dense0_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout0_p=0.45,
                     
                     dense1_num_units=600,
                     dense1_W=HeNormal(),
                     dense1_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout1_p=0.3,

                     dense2_num_units=400,
                     dense2_W=HeNormal(),
                     dense2_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout2_p=0.2,
                     
                     dense3_num_units=300,
                     dense3_W=HeNormal(),
                     dense3_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout3_p=0.1,
                                          
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,
                     
                     update=adagrad,
                     update_learning_rate=shared(float32(0.01)),

                     batch_iterator_train=BatchIterator(batch_size=256),
                     on_epoch_finished=[
                     AdjustVariable('update_learning_rate', start=0.01,
                                    stop=0.005),
#                     AdjustVariable('update_momentum', start=0.1, stop=0.0),
                     self.oneone_stopping,
                     ],
                     eval_size=0.,
                     verbose=1,
                     max_epochs=400                     
                     )
        self.nnt = deepcopy(self.nn)
        
        self.nn2 = My_NeuralNet(layers=self.layers,                     
                     input_shape=(None, self.num_features),
                     dropouti_p=0.12,
                     
                     dense0_num_units=900,
                     dense0_W=HeNormal(),
                     dense0_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout0_p=0.47,
                     
                     dense1_num_units=600,
                     dense1_W=HeNormal(),
                     dense1_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout1_p=0.32,

                     dense2_num_units=400,
                     dense2_W=HeNormal(),
                     dense2_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout2_p=0.22,
                     
                     dense3_num_units=300,
                     dense3_W=HeNormal(),
                     dense3_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout3_p=0.12,
                                          
                     output_num_units=self.num_classes,
                     output_nonlinearity=softmax,
                     
                     update=adagrad,
                     update_learning_rate=shared(float32(0.01)),

                     batch_iterator_train=BatchIterator(batch_size=256),
                     on_epoch_finished=[
                     AdjustVariable('update_learning_rate', start=0.01,
                                    stop=0.005),
#                     AdjustVariable('update_momentum', start=0.1, stop=0.0),
#                     self.early_stopping2,
                     self.oneone_stopping2,
                     ],
                     eval_size=0.,
                     verbose=1,
                     max_epochs=400                     
                     ) 
        self.nn2t = deepcopy(self.nn2)  
           
        self.nn3 = My_NeuralNet(layers=self.layers,                     
                     input_shape=(None, self.num_features),
                     dropouti_p=0.14,
                     
                     dense0_num_units=900,
                     dense0_W=HeNormal(),
                     dense0_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout0_p=0.49,
                     
                     dense1_num_units=600,
                     dense1_W=HeNormal(),
                     dense1_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout1_p=0.34,

                     dense2_num_units=400,
                     dense2_W=HeNormal(),
                     dense2_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout2_p=0.24,
                     
                     dense3_num_units=300,
                     dense3_W=HeNormal(),
                     dense3_nonlinearity = LeakyRectify(leakiness=0.05),
                     dropout3_p=0.14,
                                          
                     output_num_units=self.num_classes,
                     output_nonlinearity=softmax,
                     
                     update=adagrad,
                     update_learning_rate=shared(float32(0.01)),

                     batch_iterator_train=BatchIterator(batch_size=256),
                     on_epoch_finished=[
                     AdjustVariable('update_learning_rate', start=0.01,
                                    stop=0.005),
#                     AdjustVariable('update_momentum', start=0.1, stop=0.0),
#                     self.early_stopping,
                     self.oneone_stopping3,
                     ],
                     eval_size=0.,
                     verbose=1,
                     max_epochs=400                     
                     ) 
        self.nn3t = deepcopy(self.nn3) 

           
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        self.nn.max_epochs = 300
        self.nn.verbose=1
        self.nn.fit(X_train, y_train, X_valid, y_valid)
        
        params = self.nn.get_all_params_values()
        
        self.nn2.load_params_from(params)
        
        self.nn2.fit(X_train, y_train, X_valid, y_valid)
        
        params2 = self.nn2.get_all_params_values()
        
        self.nn3.load_params_from(params2)
        
        self.nn3.fit(X_train, y_train, X_valid, y_valid)        
        
        yp0 = self.nn3.predict_proba(X_valid)
        print 'Nolearn log-loss: %s'%(logloss_mc(y_valid, yp0))
        
        y_pred = yp0
        
#        pdb.set_trace()
        
        return y_pred
        
    def train_test(self, X, y, X_test):
        """
        """

        self.nnt.max_epochs = self.oneone_stopping.best_valid_epoch
        self.nnt.fit(X, y)
        
        params = self.nnt.get_all_params_values()
        
        self.nn2t.load_params_from(params)
        self.nn2t.max_epochs = self.oneone_stopping2.best_valid_epoch
        self.nn2t.fit(X, y)
        
        params2 = self.nn2t.get_all_params_values()
        
        self.nn3t.load_params_from(params2)
        
        self.nn3t.max_epochs = self.oneone_stopping3.best_valid_epoch
        
        self.nn3t.fit(X, y)
        
        yp0 = self.nn3t.predict_proba(X_test)
#        self.cal_clf.fit(X, y)        
#        yp1 = self.cal_clf.predict_proba(X_test)
#        y_pred = (yp0 + yp1)/2.
        y_pred = yp0
        return y_pred              
        
        

        

        

        