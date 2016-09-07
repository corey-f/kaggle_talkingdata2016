# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import math


print "Preparing tavel/time df..."


# Read in the data
events = pd.read_csv("data/events.csv", dtype={'device_id': np.str})
time_df = events[['device_id', 'timestamp', 'event_id']].reset_index(drop=True)


# prepare time dataframe
timestamp_series_object = pd.DatetimeIndex(time_df['timestamp'])
time_df['event_count'] = time_df.groupby(['device_id'])['event_id'].transform('count')
time_df['event_count'] = time_df['event_count'] / (1.0 * time_df['event_count'].max())
time_df['hour'] = timestamp_series_object.hour
time_df = pd.get_dummies(time_df, prefix='hour', columns=['hour'])
time_df.drop(['timestamp', 'event_id'], axis=1, inplace=True)
time_df = time_df.groupby(['device_id']).mean().reset_index()
devices_with_data = time_df
devices_with_data.fillna(0.0, inplace=True)


# Verify and save to disk
print devices_with_data.shape
devices_with_data.to_csv("dense_features_df.csv", index=False)
print "done"
