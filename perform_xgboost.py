import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import log_loss
import time
import datetime
from scipy.sparse import csr_matrix
from scipy.sparse import hstack


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


# Fit, validate, and make predictions with an XGBoost model
def run_xgb(X_full_train, y_full_train, X_test, random_state=1337):
    eta = 0.3
    max_depth = 3 
    subsample = 0.7 
    colsample_bytree = 0.6 
    min_child_weight = 1.0 
    gamma = 1.0
    start_time = time.time()

    X_full_train = hstack((X_full_train, csr_matrix(np.ones((X_full_train.shape[0], 1)))))
    X_test = hstack((X_test, csr_matrix(np.ones((X_test.shape[0], 1)))))

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "multi:softprob",
        "num_class": 12,
        "booster" : "gbtree",
        "eval_metric": "mlogloss",
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 1500
    early_stopping_rounds = 50
    test_size = 0.3

    X_train, X_valid, y_train, y_valid = train_test_split(X_full_train, y_full_train, test_size=test_size, random_state=random_state)
    print 'Length train:', X_train.shape[0]
    print 'Length valid:', X_valid.shape[0]
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration)
    score = log_loss(y_valid[y_valid.columns[0]].tolist(), check)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


# Read in the train/test feature data
X_train = load_sparse_csr("X_train.csr.npz")
X_test = load_sparse_csr("X_test.csr.npz")
y_train = pd.read_csv("y_train.csv")
test_device_ids = pd.read_csv("y_test_device_ids.csv", dtype={'device_id': np.str})
full_test_df = pd.read_csv("data/gender_age_test.csv", dtype={'device_id': np.str})
test_device_ids = full_test_df['device_id']
test_device_ids_df = pd.DataFrame(test_device_ids)
test_device_ids_df.columns = ['device_id']


# Output verification info
print 'Length of train: ', X_train.shape[0]
print 'Length of y train: ', len(y_train)
print 'Length of test: ', X_test.shape[0]


# Fit, validate, and make predictions with an XGBoost model
test_prediction, score = run_xgb(X_train, y_train, X_test)
print("LS: {}".format(round(score, 5)))


# Prepare the submission
test_prediction_df = pd.DataFrame(test_prediction)
print test_prediction_df.shape
print test_device_ids_df.shape
submission_df = pd.concat([test_device_ids_df, test_prediction_df], axis=1)
reordered_columns = ['device_id'] + range(12)
submission_df = submission_df[reordered_columns]
submission_df.columns = ['device_id','F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
this_filename = 'XGB_submission_df' + str(score) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv"
submission_df.to_csv(this_filename, index=False)
