import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


N_ROWS = None


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

# Read in dense and sparse feature data
sparse_features_df = pd.read_csv("sparse_features_df.csv", dtype={'device_id': np.str}, nrows=N_ROWS)
dense_features_df = pd.read_csv("dense_features_df.csv", dtype={'device_id': np.str})
dense_cols_list = list(dense_features_df.columns)

# Verify input shapes
print sparse_features_df.shape
print dense_features_df.shape

# Add in a binary flag feature for the device ids that do not have events
train_data_df = pd.read_csv("data/gender_age_train.csv", dtype={'device_id':np.str}, nrows=N_ROWS)
train_data_df.drop(["gender", "age", "group"], axis=1, inplace=True)
test_data_df = pd.read_csv("data/gender_age_test.csv", dtype={'device_id':np.str}, nrows=N_ROWS)
events_df = pd.read_csv("data/events.csv", dtype={'device_id':np.str}, nrows=N_ROWS)
events_df.drop_duplicates('device_id', keep='first', inplace=True)
has_events_df = pd.concat((train_data_df, test_data_df), axis=0, ignore_index=True)
has_events_df.drop_duplicates('device_id', keep='first', inplace=True)
has_events_df = has_events_df.merge(events_df, how='left', on='device_id')
has_events_df.drop(["timestamp", "longitude", "latitude"], axis=1, inplace=True)
has_events_df = has_events_df[pd.isnull(has_events_df['event_id'])].reset_index(drop=True)
has_events_df.drop(["event_id"], axis=1, inplace=True)
has_events_df['feature'] = "missing_events"
has_events_df['value'] = 1.0
sparse_features_df = pd.concat((sparse_features_df, has_events_df), axis=0, ignore_index=True)


# Translate dense features to sparse format
for this_dense_feature in dense_cols_list:
    if this_dense_feature != 'device_id':
        this_df = dense_features_df[['device_id', this_dense_feature]].reset_index()
        this_df['feature'] = this_dense_feature
        this_df = this_df[['device_id', 'feature', this_dense_feature]]
        this_df.columns = [this_df.columns[0], this_df.columns[1], 'value']
        sparse_features_df = pd.concat((sparse_features_df, this_df), axis=0, ignore_index=True)
print sparse_features_df.shape


# Create the sparse matrix
num_device_ids = len(sparse_features_df["device_id"].unique())
num_features = len(sparse_features_df["feature"].unique())
device_id_encoder = LabelEncoder().fit(sparse_features_df["device_id"])
row_ids = device_id_encoder.transform(sparse_features_df["device_id"])
col_ids = LabelEncoder().fit_transform(sparse_features_df["feature"])
value_data = sparse_features_df["value"]
sparse_matrix = csr_matrix( (value_data, (row_ids, col_ids)), shape=(num_device_ids, num_features) )
print sparse_matrix.shape


# Read in the train/test data
training_df = pd.read_csv('data/gender_age_train.csv', dtype={'device_id': np.str})
training_df.drop(["age", "gender"], axis=1, inplace=True)
testing_df = pd.read_csv('data/gender_age_test.csv', dtype={'device_id': np.str})


# Prepare the formatted train/test data features
y_train = training_df["group"]
target_group_encoder = LabelEncoder()
y_train = target_group_encoder.fit_transform(y_train)
y_train = pd.DataFrame(y_train)
y_train.columns = ['target_category']
test_device_ids = pd.DataFrame(testing_df["device_id"])
test_device_ids.columns = ['device_id']
training_rows = device_id_encoder.transform(training_df["device_id"])
X_train = sparse_matrix[training_rows, :]
testing_rows = device_id_encoder.transform(testing_df["device_id"])
X_test = sparse_matrix[testing_rows, :]


# Verify the train/test feature data
print "x train"
print X_train.shape

print "x test"
print X_test.shape

print "test_device_ids"
print test_device_ids.shape

print "y_train"
print y_train.shape


# Save the train/test feature data to disk
save_sparse_csr("X_train.csr", X_train)
save_sparse_csr("X_test.csr", X_test)
test_device_ids.to_csv("y_test_device_ids.csv", index=False)
y_train.to_csv("y_train.csv", index=False)


