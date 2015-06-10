"""
"""
import sys
import numpy as np

#sys.path.append('/home/sandrovegapons/anaconda/src/xgboost/wrapper')
sys.path.append('E:\Competitions\OttoGroup\py_ml_utils\lib')
from xgboost import Booster 


from classifier import Clf

if sys.version_info[0] == 3:
    string_types = str,
else:
    string_types = basestring,


def my_train_xgboost(params, dtrain, num_boost_round=10, evals=(), obj=None, 
                     feval=None, early_stopping_rounds=None, seed=0, 
                     rt_eta=1.0006, rt_ssp=1.0006, rt_clb=1.0006, 
                     rt_dpt=1.0001):
    """
    Train a booster with given parameters.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained.
    num_boost_round: int
        Number of boosting iterations.
    watchlist : list of pairs (DMatrix, string)
        List of items to be evaluated during training, this allows user to watch
        performance on the validation set.
    obj : function
        Customized objective function.
    feval : function
        Customized evaluation function.
    early_stopping_rounds: int
        Activates early stopping. Validation error needs to decrease at least
        every <early_stopping_rounds> round(s) to continue training.
        Requires at least one item in evals.
        If there's more than one, will use the last.
        Returns the model from the last iteration (not the best one).
        If early stopping occurs, the model will have two additional fields:
        bst.best_score and bst.best_iteration.

    Returns
    -------
    booster : a trained booster model
    """
    eta = params['eta']   

    ssp = params['subsample']
    clb = params['colsample_bytree']
    
#    rt_eta=np.random.random()
    rt_ssp=np.random.uniform(0.1,0.9)
    rt_clb=np.random.uniform(0.1,0.9)
    

    evals = list(evals)
    bst = Booster(params, [dtrain] + [d[0] for d in evals], seed=seed)

    if not early_stopping_rounds:
        for i in range(num_boost_round):
            bst.set_param({'eta': eta})
            bst.set_param({'subsample': ssp})
            bst.set_param({'colsample_bytree': clb})
            eta = eta * rt_eta
#            ssp = ssp * rt_ssp
#            clb = clb * rt_clb
            ssp = rt_ssp
            clb = rt_clb
            bst.update(dtrain, i, obj)
            if len(evals) != 0:
                bst_eval_set = bst.eval_set(evals, i, feval)
                if isinstance(bst_eval_set, string_types):
                    sys.stderr.write(bst_eval_set + '\n')
                else:
                    sys.stderr.write(bst_eval_set.decode() + '\n')
        return bst

    else:
        # early stopping

        if len(evals) < 1:
            raise ValueError('For early stopping you need at least on set in evals.')

        sys.stderr.write("Will train until {} error hasn't decreased in {} rounds.\n".format(evals[-1][1], early_stopping_rounds))

        # is params a list of tuples? are we using multiple eval metrics?
        if type(params) == list:
            if len(params) != len(dict(params).items()):
                raise ValueError('Check your params. Early stopping works with single eval metric only.')
            params = dict(params)

        # either minimize loss or maximize AUC/MAP/NDCG
        maximize_score = False
        if 'eval_metric' in params:
            maximize_metrics = ('auc', 'map', 'ndcg')
            if filter(lambda x: params['eval_metric'].startswith(x), maximize_metrics):
                maximize_score = True

        if maximize_score:
            best_score = 0.0
        else:
            best_score = float('inf')

        best_msg = ''
        best_score_i = 0

        for i in range(num_boost_round):
            bst.set_param({'eta': eta})
            bst.set_param({'subsample': ssp})
            bst.set_param({'colsample_bytree': clb})
            eta = eta * rt_eta
#            ssp = ssp * rt_ssp
#            clb = clb * rt_clb
            ssp = rt_ssp
            clb = rt_clb
            bst.update(dtrain, i, obj)
            bst_eval_set = bst.eval_set(evals, i, feval)

            if isinstance(bst_eval_set, string_types):
                msg = bst_eval_set
            else:
                msg = bst_eval_set.decode()

            sys.stderr.write(msg + '\n')

            score = float(msg.rsplit(':', 1)[1])
            if (maximize_score and score > best_score) or \
                    (not maximize_score and score < best_score):
                best_score = score
                best_score_i = i
                best_msg = msg
            elif i - best_score_i >= early_stopping_rounds:
                sys.stderr.write("Stopping. Best iteration:\n{}\n\n".format(best_msg))
                bst.best_score = best_score
                bst.best_iteration = best_score_i
                return bst

        return bst


class Clf_xgboost(Clf):
    """
    Base class for xgboost based classifiers.
    """
    pass
        