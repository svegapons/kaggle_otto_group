"""
"""
import numpy as np
from time import time

from classifier import Clf
from nolearn.lasagne import NeuralNet
import pdb

from sklearn.preprocessing import LabelEncoder

class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'
        
                     
def float32(k):
    """
    """
    return np.cast['float32'](k)
  
  
class AdjustVariable(object):
    """
    """
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value) 

            
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
#            pdb.set_trace()
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()            


class OneOneStopping(object):
    def __init__(self, ratio=0.97):
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None
        self.ratio = ratio

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_train = train_history[-1]['train_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif current_train / current_valid < self.ratio :
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()     

            
class My_NeuralNet(NeuralNet):
    """
    Simple modification of nolearn NeuralNet to allow fixed validation set on 
    training.
    """
#    def __init__(self):
#        super(My_NeuralNet, self).__init__()
        
    def fit(self, X, y, X_valid=[], y_valid=[]):
        if self.use_label_encoder:
            self.enc_ = LabelEncoder()
            y = self.enc_.fit_transform(y).astype(np.int32)
            self.classes_ = self.enc_.classes_
#        pdb.set_trace()
#        print self._output_layer
        self._initialized = False
        self.initialize()
        self.train_history_ = []
        
        

        try:
            self.train_loop(X, y, X_valid, y_valid)
        except KeyboardInterrupt:
            pass
        return self
        
        
    def train_loop(self, X, y, X_valid=[], y_valid=[]):
#        pdb.set_trace()
        if (not len(X_valid) == 0) and (not len(y_valid) == 0):
            X_train = X
            y_train = y
            
        else:
            X_train, X_valid, y_train, y_valid = self.train_test_split(
                X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        first_iteration = True
        num_epochs_past = len(self.train_history_)

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            custom_score = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)
                if self.custom_score:
                    y_prob = self.predict_iter_(Xb)
                    custom_score.append(self.custom_score[1](yb, y_prob))

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if custom_score:
                avg_custom_score = np.mean(custom_score)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                }
            if self.custom_score:
                info[self.custom_score[0]] = avg_custom_score
            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

    
            
            
class Clf_nolearn(Clf):
    """
    Base class for nolearn based classifiers.
    """
    def __init__(self, num_features, num_classes):
        """
        """        
        self.layers = None    
        self.num_features = num_features
        self.num_classes = num_classes        
        
        
                     


        


        

        