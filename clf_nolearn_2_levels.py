"""
"""
import numpy as np

from nolearn.lasagne import BatchIterator

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.nonlinearities import LeakyRectify, softmax
from lasagne.init import HeNormal
from lasagne.updates import adagrad

from theano import shared

from sklearn.preprocessing import LabelEncoder

from clf_nolearn import Clf_nolearn, My_NeuralNet, EarlyStopping, AdjustVariable, float32


class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'
    
            
class Clf_nolearn_2_levels(Clf_nolearn):
    """
    Simple nolearn based classifier...
    """
    def __init__(self, num_features=93, num_classes=9):
        """
        """
        self.prefix = 'nl_2lv'
        self.num_features = num_features
        self.num_classes = num_classes 
        
        self.layers0 = [('input', InputLayer),
                   ('dropoutn', DropoutLayer),
                   ('dense0', DenseLayer),
                   ('dropout0', DropoutLayer),
                   ('dense1', DenseLayer),
                   ('dropout1', DropoutLayer),
                   ('dense2', DenseLayer),
                   ('dropout2', DropoutLayer),
                   ('dense3', DenseLayer),
                   ('dropout3', DropoutLayer),
                   ('output', DenseLayer)]
    
        self.early_stopping0 = EarlyStopping(patience=20)
        
        self.nn0 = My_NeuralNet(layers=self.layers0,                 
                 input_shape=(None, self.num_features),
                 dropoutn_p=0.12,
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
                                  
                 output_num_units=2,
                 output_nonlinearity=softmax,
                 
                 update=adagrad,
                 update_learning_rate=shared(float32(0.01)),
                 
                 batch_iterator_train=BatchIterator(batch_size=64),
                 on_epoch_finished=[
                 AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
#                     AdjustVariable('update_momentum', start=0.1, stop=0.0),
                 self.early_stopping0,
                 ],
                 eval_size=0,
                 verbose=1,
                 max_epochs=36                 
                 )
    

        self.layers1 = [('input', InputLayer),
                   ('dropoutn', DropoutLayer),
                   ('dense0', DenseLayer),
    #               ('gaussian0', GaussianNoiseLayer),
                   ('dropout0', DropoutLayer),
                   ('dense1', DenseLayer),
    #               ('gaussian1', GaussianNoiseLayer),
                   ('dropout1', DropoutLayer),
                   ('dense2', DenseLayer),
                   ('dropout2', DropoutLayer),
    #               ('dense3', DenseLayer),
    #               ('dropout3', DropoutLayer),
                   ('output', DenseLayer)]
                   
        self.early_stopping1 = EarlyStopping(patience=20)    
        
        self.nn1 = My_NeuralNet(layers=self.layers1,
                         
                         input_shape=(None, self.num_features),
                         dropoutn_p=0.18,
                         dense0_num_units=900,
                         dense0_W=HeNormal(),
                         dense0_nonlinearity = LeakyRectify(leakiness=0.002),
                         dropout0_p=0.35,
                         
                         dense1_num_units=600,
                         dense1_W=HeNormal(),
                         dense1_nonlinearity = LeakyRectify(leakiness=0.002),
                         dropout1_p=0.15,
    
                         dense2_num_units=400,
                         dense2_W=HeNormal(),
                         dense2_nonlinearity = LeakyRectify(leakiness=0.002),    
                         dropout2_p=0.05,

                         output_num_units=3,
                         output_nonlinearity=softmax,
                         
                         update=adagrad,
                         update_learning_rate=shared(float32(0.01)),
                         
                         batch_iterator_train=BatchIterator(batch_size=64),
                         on_epoch_finished=[
                         AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
    #                     AdjustVariable('update_momentum', start=0.1, stop=0.0),
                         self.early_stopping1,
                         ],
                         eval_size=0,
                         verbose=1,
                         max_epochs=60                         
                         )        
    
        self.layers2 = [('input', InputLayer),
                   ('dropoutn', DropoutLayer),
                   ('dense0', DenseLayer),
    #               ('gaussian0', GaussianNoiseLayer),
                   ('dropout0', DropoutLayer),
                   ('dense1', DenseLayer),
    #               ('gaussian1', GaussianNoiseLayer),
                   ('dropout1', DropoutLayer),
                   ('dense2', DenseLayer),
                   ('dropout2', DropoutLayer),
    #               ('dense3', DenseLayer),
    #               ('dropout3', DropoutLayer),
                   ('output', DenseLayer)]    
                   
        self.early_stopping2 = EarlyStopping(patience=20)
        
        self.nn2 = My_NeuralNet(layers=self.layers2,
                         
                         input_shape=(None, self.num_features),
                         dropoutn_p=0.13,
                         dense0_num_units=900,
                         dense0_W=HeNormal(),
                         dense0_nonlinearity = LeakyRectify(leakiness=0.002),
                         dropout0_p=0.35,
                         
                         dense1_num_units=600,
                         dense1_W=HeNormal(),
                         dense1_nonlinearity = LeakyRectify(leakiness=0.002),
                         dropout1_p=0.15,
    
                         dense2_num_units=400,
                         dense2_W=HeNormal(),
                         dense2_nonlinearity = LeakyRectify(leakiness=0.002),    
                         dropout2_p=0.05,
                         
                         output_num_units=6,
                         output_nonlinearity=softmax,
                         
                         update=adagrad,
                         update_learning_rate=shared(float32(0.01)),

                         
                         batch_iterator_train=BatchIterator(batch_size=64),
                         on_epoch_finished=[
                         AdjustVariable('update_learning_rate', start=0.01, stop=0.001),
    #                     AdjustVariable('update_momentum', start=0.1, stop=0.0),
                         self.early_stopping2,
                         ],
    #                     eval_size=0.2,
                         eval_size=0,
                         verbose=1,
                         max_epochs=31
                         
                         )
                     

    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        le = LabelEncoder()
        id_123 = np.logical_or(np.logical_or(y_train==1, y_train==2), 
                               y_train==3)  
        y0 = np.zeros(len(y_train), dtype=np.int32)
        y0[id_123] = 1
        X0 = np.copy(X_train) 
        y0 = le.fit_transform(y0).astype(np.int32)
    
        X1 = X_train[id_123]
        y1 = y_train[id_123]
        y1 = le.fit_transform(y1).astype(np.int32)
    
        X2 = X_train[np.logical_not(id_123)]
        y2 = y_train[np.logical_not(id_123)]    
        y2 = le.fit_transform(y2).astype(np.int32)    
        
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
        
        self.nn0.max_epochs = 300
        self.nn0.verbose=1
        self.nn0.fit(X0, y0, X0_valid, y0_valid)
        y0_pred = self.nn0.predict_proba(X_valid)
        
        self.nn1.max_epochs = 300
        self.nn1.verbose=1
        self.nn1.fit(X1, y1, X1_valid, y1_valid)
        y1_pred = self.nn1.predict_proba(X_valid)        

        self.nn2.max_epochs = 300
        self.nn2.verbose=1
        self.nn2.fit(X2, y2, X2_valid, y2_valid)
        y2_pred = self.nn2.predict_proba(X_valid)
           
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
        le = LabelEncoder()
        id_123 = np.logical_or(np.logical_or(y==1, y==2), y==3)  
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
        
        print 'working on nn0...'
        self.nn0.max_epochs = self.early_stopping0.best_valid_epoch
        self.nn0.verbose=0
        self.nn0.fit(X0, y0)
        y0_pred = self.nn0.predict_proba(X_test)
        
        print 'working on nn1...'
        self.nn1.max_epochs = self.early_stopping1.best_valid_epoch
        self.nn1.verbose=0
        self.nn1.fit(X1, y1)
        y1_pred = self.nn1.predict_proba(X_test)   
        
        print 'working on nn2...'
        self.nn2.max_epochs = self.early_stopping2.best_valid_epoch
        self.nn2.verbose=0        
        self.nn2.fit(X2, y2)
        y2_pred = self.nn2.predict_proba(X_test)
           
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
        


        

        