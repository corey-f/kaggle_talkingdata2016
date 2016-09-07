import numpy as np
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt


# Prepare plot data
gender_age_train_df = pd.read_csv("data/gender_age_train.csv", dtype={'device_id': np.str}, usecols=['device_id', 'gender', 'age'])
events_df = pd.read_csv("data/events.csv", dtype={'device_id': np.str}, usecols=['device_id', 'timestamp'], nrows=100000)
plot_data = events_df.merge(gender_age_train_df, how='inner', on='device_id')
timestamp_series_object = pd.DatetimeIndex(plot_data['timestamp'])
plot_data['hour'] = timestamp_series_object.hour
print "Number of unique device ids: {}".format(len(plot_data['device_id'].unique()))
plot_data.drop(['device_id', 'timestamp'], axis=1, inplace=True)
print len(plot_data)
print plot_data.head(5)


# Show violin plot of user age/gender by hour
sns.set(style="whitegrid", palette="muted", font_scale=2.5)
fig1 = plt.figure()
sns.violinplot(x="hour", y="age", hue="gender", data=plot_data, size=10, palette="muted", scale="count", split=True)
plt.show()

