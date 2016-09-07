import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import datetime
from scipy.sparse import csr_matrix
from sklearn.grid_search import GridSearchCV


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


# Score a classifier using cross validation
def score(clf, this_x, this_y, random_state = 1337):
    this_y_series = this_y[this_y.columns[0]]
    nclasses = len(this_y_series.unique())
    print "Number of classes: {}".format(nclasses)
    kf = StratifiedKFold(this_y_series, n_folds=5, shuffle=True, random_state=random_state)
    pred = np.zeros((this_y.shape[0], nclasses))
    for i_train, i_test in kf:
        this_x_train, this_x_test = this_x[i_train, :], this_x[i_test, :]
        this_y_train, this_y_test = this_y_series[i_train], this_y_series[i_test]
        clf.fit(this_x_train, this_y_train)
        pred[i_test,:] = clf.predict_proba(this_x_test)
        this_log_loss = log_loss(this_y_test, pred[i_test, :])
        print "StratifiedKFold iteration log loss: {:.5f}".format(this_log_loss)
    print('')
    this_log_loss = log_loss(this_y_series, pred)
    print "StratifiedKFold log loss: {:.5f}".format(this_log_loss)
    return this_log_loss


# Load the train/test feature data
X_train = load_sparse_csr("X_train.csr.npz")
X_test = load_sparse_csr("X_test.csr.npz")
y_train = pd.read_csv("y_train.csv")
test_device_ids = pd.read_csv("y_test_device_ids.csv", dtype={'device_id': np.str})


# Fit a LR model
lrc = LogisticRegression(multi_class='multinomial', solver='newton-cg', max_iter=1000)
parameters = {'C':[.001, .01, .1, 1, 10, 100], 'solver':['newton-cg', 'lbfgs']}
clf = GridSearchCV(lrc, parameters)
clf.fit(X_train, y_train[y_train.columns[0]])


# Output score and params
print "Best score: {}".format(clf.best_score_)
print "Choosen Parameters:"
print clf.best_params_


# Prepare submission
submission_columns = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
submission_df = pd.DataFrame(clf.predict_proba(X_test), index = test_device_ids.values, columns=submission_columns)
submission_df.index.name = 'device_id'
submission_df.head(5)
this_filename = 'LR_submission_df' + str(this_score) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv"
submission_df.to_csv(this_filename, index=True)

