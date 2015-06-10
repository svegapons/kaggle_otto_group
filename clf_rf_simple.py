"""
"""
import sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from classifier import Clf, logloss_mc



class Clf_rf_simple(Clf):
    """
    Simple xgboost based classifier.
    """    
    def __init__(self, num_classes=9):
        """
        """
        self.prefix = 'rf_simple'
    
    def train_validate(self, X_train, y_train, X_valid, y_valid):
        """
        """
        rf = RandomForestClassifier(n_estimators=1500, 
                                    class_weight='auto', max_features=0.8)
        rf.fit(X_train, y_train)
        yp0 = rf.predict_proba(X_valid)
        print logloss_mc(y_valid, yp0)
        
        rf = RandomForestClassifier(n_estimators=1500, 
                                    class_weight='auto', max_features=0.8)
        
        cc = CalibratedClassifierCV(base_estimator=rf, method='isotonic',
                                    cv=StratifiedKFold(y_train, 3))
        cc.fit(X_train, y_train)
        yp1 = cc.predict_proba(X_valid)
        print logloss_mc(y_valid, yp1)
        
        y_pred = (yp0 + yp1)/2.

        return y_pred
        
        
    def train_test(self, X, y, X_test):
        """
        """
        rf = RandomForestClassifier(n_estimators=1500,
                                    class_weight='auto', max_features=0.8)
                                    
        rf.fit(X, y)
        yp0 = rf.predict_proba(X_test)
        rf = RandomForestClassifier(n_estimators=1500, 
                            class_weight='auto', max_features=0.8)
                            
        cc = CalibratedClassifierCV(base_estimator=rf, method='isotonic',
                                    cv=StratifiedKFold(y, 3))
        cc.fit(X, y)
        yp1 = cc.predict_proba(X_test)
        
        y_pred = (yp0 + yp1)/2.
        
        return y_pred               
        