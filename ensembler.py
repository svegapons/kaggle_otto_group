"""
"""
import os
import numpy as np
import pdb

    

class Ensembler:
    """
    Base class for our ensemblers...
    """
       
    def process(self, validation_folder='./validation', test_folder='./test',
                y_valid_path='./data/y_valid.csv', verbose=1):
        """
        """
        lv_path = os.listdir(validation_folder)
        lv_path.sort()
        print lv_path
        list_valid = []        
        for p in lv_path:
            arr = np.loadtxt(os.path.join(validation_folder, p))
            list_valid.append(arr)
        
        self.n_preds = len(list_valid)
        
        X = np.hstack(list_valid)
        y = np.loadtxt(y_valid_path)
        
#        pdb.set_trace()
        lt_path = os.listdir(test_folder)
        lt_path.sort()
        list_test = []        
        for p in lt_path:
            arr = np.loadtxt(os.path.join(test_folder, p))
            list_test.append(arr)
        
        X_test = np.hstack(list_test)
        
        y_pred = self.internal_processing(X, y, X_test)
        
        return y_pred
                           
        
    def internal_processing(self, X, y, X_test):
        pass
    
        
    


    