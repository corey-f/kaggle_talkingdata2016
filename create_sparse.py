import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


N_ROWS = None


app_labels = pd.read_csv("data/app_labels.csv", dtype={'app_id': np.str}, nrows=N_ROWS)
app_events = pd.read_csv("data/app_events.csv", dtype={'app_id': np.str}, nrows=N_ROWS)  # , nrows=12646)
map_event_to_device_id = pd.read_csv("data/events.csv", usecols=['event_id', 'device_id'], index_col='event_id', dtype={'device_id': np.str})['device_id']
phone_device_data = pd.read_csv('data/phone_brand_device_model.csv', dtype={'device_id': np.str}, encoding='utf-8')


# Prepare app category features
app_cats = app_events.merge(app_labels, how='left', on='app_id')
app_cats = app_cats.dropna()
app_cats["label_id"] = app_cats["label_id"].apply(lambda x: "cat_" + str(x))
app_cats["value"] = app_cats["is_installed"]  + ( 2.0 * app_cats["is_active"] )
app_cats["device_id"] = app_cats["event_id"].map(map_event_to_device_id)
app_cats = app_cats[["device_id", "label_id", "value"]]
app_cats['value'] = app_cats.groupby(['device_id','label_id'])['value'].transform('mean')
app_cats.drop_duplicates(['device_id','label_id'], keep='first', inplace=True)
app_cats.columns = [app_cats.columns[0], 'feature', app_cats.columns[2]]
f1 = app_cats.reset_index(drop=True)


# Prepare app features
app_events["device_id"] = app_events["event_id"].map(map_event_to_device_id)
app_events['value'] = app_events["is_installed"]  + ( 2.0 * app_events["is_active"] )
app_events = app_events[["device_id", "app_id", "value"]]
app_events['value'] = app_events.groupby(['device_id', 'app_id'])['value'].transform('mean')
app_events.drop_duplicates(['device_id', 'app_id'], keep='first', inplace=True)
app_events["app_id"] = app_events["app_id"].apply(lambda x: "app_" + str(x))
app_events.columns = [app_events.columns[0], 'feature', app_events.columns[2]]
f2 = app_events.reset_index(drop=True)

# Prepare brand features
brands = phone_device_data[['device_id', 'phone_brand']].reset_index(drop=True)
brands['phone_brand'] = LabelEncoder().fit_transform(brands['phone_brand'])
brands["value"] = 1.0
brands['value'] = brands.groupby(['device_id', 'phone_brand'])['value'].transform('mean')
brands.drop_duplicates(['device_id', 'phone_brand'], keep='first', inplace=True)
brands["phone_brand"] = brands["phone_brand"].apply(lambda x: "brand_" + str(x))
brands.columns = [brands.columns[0], 'feature', brands.columns[2]]
f3 = brands.reset_index(drop=True)


# Prepare device-model features
device_models = phone_device_data[['device_id', 'device_model']].reset_index(drop=True)
device_models['device_model'] = LabelEncoder().fit_transform(device_models['device_model'])
device_models["value"] = 1.0
device_models['value'] = device_models.groupby(['device_id', 'device_model'])['value'].transform('mean')
device_models.drop_duplicates(['device_id', 'device_model'], keep='first', inplace=True)
device_models["device_model"] = device_models["device_model"].apply(lambda x: "model_" + str(x))
device_models.columns = [device_models.columns[0], 'feature', device_models.columns[2]]
f4 = device_models.reset_index(drop=True)


# Prepare placement in train/test features (the leak)
train_data_df = pd.read_csv("data/gender_age_train.csv", dtype={'app_id': np.str}, nrows=N_ROWS)
test_data_df = pd.read_csv("data/gender_age_test.csv", dtype={'app_id': np.str}, nrows=N_ROWS)
train_data_df.drop(["gender", "age", 'group'], axis=1, inplace=True)
train_data_df['feature'] = "placement"
test_data_df['feature'] = "placement"
train_data_df['value'] = (np.arange(train_data_df.shape[0]) * 100.0) / train_data_df.shape[0]
test_data_df['value'] = (np.arange(test_data_df.shape[0]) * 100.0) / test_data_df.shape[0]
f5 = train_data_df
f6 = test_data_df


# Prepare a desnse list of the sparse features including device_id, feature, and value
sparse_features_df = pd.concat((f1, f2, f3, f4, f5, f6), axis=0, ignore_index=True)


# Verify and save to disk
print sparse_features_df.shape
print len(sparse_features_df)
sparse_features_df.to_csv("sparse_features_df.csv", index=False)
