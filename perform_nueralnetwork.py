import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import datetime
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


# Load the train/test feature data
X_train = load_sparse_csr("X_train.csr.npz")
X_test = load_sparse_csr("X_test.csr.npz")
y_train = pd.read_csv("y_train.csv")
test_device_ids = pd.read_csv("y_test_device_ids.csv", dtype={'device_id': np.str})
y_train = np.matrix(y_train.values)


# Split out validation data as necessary
print("# Split data")
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.999, random_state=1337)
print("# Num of Features: ", X_train.shape[1])


# create NN model
def baseline_model():
    model = Sequential()
    model.add(Dense(200, input_dim=X_train.shape[1], init='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


# Create and fit the NN model
model=baseline_model()
this_batch_size = 400
fit= model.fit_generator(generator=batch_generator(X_train, y_train, this_batch_size, True),
                         nb_epoch=35,
                         samples_per_epoch=(int(X_train.shape[0]/this_batch_size) * this_batch_size),
                         validation_data=(X_val.todense(), y_val), verbose=2
                         )


# evaluate the model
scores_val = model.predict_generator(generator=batch_generatorp(X_val, 400, False), val_samples=X_val.shape[0])
this_score = log_loss(y_val, scores_val)
print 'logloss val {}'.format(this_score)


# Prepare the submission
print "# Final prediction"
scores = model.predict_generator(generator=batch_generatorp(X_test, 800, False), val_samples=X_test.shape[0])
submission_columns = ['F23-','F24-26','F27-28','F29-32','F33-42','F43+','M22-','M23-26','M27-28','M29-31','M32-38','M39+']
result = pd.DataFrame(scores , columns=submission_columns)
result["device_id"] = test_device_ids
print(result.head(1))
result = result.set_index("device_id")
this_filename = 'NN_submission_df' + str(this_score) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + ".csv"
result.to_csv(this_filename, index=True, index_label='device_id')
print("Done")
