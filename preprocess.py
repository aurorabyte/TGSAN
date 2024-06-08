import numpy as np
import pandas as pd
import datetime as dt
import time
import json
from sklearn.impute import KNNImputer

def duration_calc(dataframe):
    diff_min_list = []
    for i in range(len(dataframe)):
        start = dt.datetime.strptime(dataframe['start_time'].iloc[i], '%M:%S')
        end = dt.datetime.strptime(dataframe['end_time'].iloc[i], '%M:%S')
        diff = (end - start)
        diff_min = diff.seconds / 60
        diff_min_list.append(diff_min)
    return diff_min_list

def conversation_duration_calculate(dataframe):
    duration_list = []
    for i in range(len(dataframe)):
        start = dt.datetime.strptime(dataframe['start_time'].iloc[i], '%M:%S')
        end = dt.datetime.strptime(dataframe['end_time'].iloc[i], '%M:%S')
        duration = end - start
        duration_list.append(duration.seconds)
    return duration_list

def final_check(dataframe):
    if len(dataframe.columns[dataframe.isna().any()].tolist()) == 0:
        return dataframe
    else:
        dataframe = dataframe.replace(np.nan, 0)
        return dataframe

def data_imputer(dataframe, column_name):
    features = dataframe[column_name]
    features = np.array(features)
    features = features.reshape(-1, 1)
    imputer = KNNImputer(n_neighbors=2, weights='distance')
    imputed_data = imputer.fit_transform(features)
    return imputed_data

def activity_preprocess(uid):
    file = './dataset/sensing/activity/activity_' + uid + '.csv'
    df_activity = pd.read_csv(file)
    df_activity['date'] = df_activity['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H", time.gmtime(x)))
    df_activity = df_activity[['date', ' activity inference']]
    df_activity = df_activity.sort_values(by='date', ascending=True)
    df_activity['activity_stationary'] = (df_activity[' activity inference'].values == 0).astype(int)
    df_activity['activity_walking'] = (df_activity[' activity inference'].values == 1).astype(int)
    df_activity['activity_running'] = (df_activity[' activity inference'].values == 2).astype(int)
    df_activity['activity_unknown'] = (df_activity[' activity inference'].values == 3).astype(int)
    df_activity = df_activity[['date', 'activity_stationary', 'activity_walking', 'activity_running', 'activity_unknown']]
    df_activity = df_activity.groupby('date', as_index=False).mean()
    df_activity.columns = ['date', 'activity_stationary_count', 'activity_walking_count', 'activity_running_count', 'activity_unknown_count']
    df_activity = final_check(df_activity)
    df_activity.to_excel('./extracting/activity/activity_' + uid + '.xlsx', index=False)

def audio_preprocess(uid):
    file = './dataset/sensing/audio/audio_' + uid + '.csv'
    df_audio = pd.read_csv(file)
    df_audio['date'] = df_audio['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d %H", time.gmtime(x)))
    df_audio = df_audio[['date', ' audio inference']]
    df_audio = df_audio.sort_values(by='date', ascending=True)
    df_audio['audio_silence'] = (df_audio[' audio inference'].values == 0).astype(int)
    df_audio['audio_voice'] = (df_audio[' audio inference'].values == 1).astype(int)
    df_audio['audio_noise'] = (df_audio[' audio inference'].values == 2).astype(int)
    df_audio['audio_unknown'] = (df_audio[' audio inference'].values == 3).astype(int)
    df_audio = df_audio[['date', 'audio_silence', 'audio_voice', 'audio_noise', 'audio_unknown']]
    df_audio = df_audio.groupby('date', as_index=False).sum()
    df_audio.columns = ['date', 'audio_silence_count', 'audio_voice_count', 'audio_noise_count', 'audio_unknown_count']
    df_audio = final_check(df_audio)
    df_audio.to_excel('./extracting/audio/audio_' + uid + '.xlsx', index=False)

def wifi_location_preprocess(uid):
    file = './dataset/sensing/wifi_location/wifi_location_' + uid + '.csv'
    df_wifi_loc_raw = pd.read_csv(file, index_col=False)
    df_wifi_loc_raw['date'] = df_wifi_loc_raw['time'].apply(lambda x: time.strftime("%Y-%m-%d %H", time.gmtime(x)))
    df_wifi_loc = df_wifi_loc_raw[['date', 'location']]
    df_wifi_loc = df_wifi_loc.sort_values(by='date', ascending=True)
    key_feature_list = []
    loc_feature_list = []
    for key, item in df_wifi_loc.groupby('date'):
        key_feature_list.append(key)
        loc_feature_list.append(len(item['location'].unique()))
    df_wifi_loc = pd.DataFrame()
    df_wifi_loc['date'] = key_feature_list
    df_wifi_loc['number of distinct locations'] = loc_feature_list
    file = './dataset_1_提取特征/activity/activity_' + uid + '.xlsx'
    np_activity = pd.read_excel(file, index_col=False).values[:, 0:2]
    complete_np_wifi_loc = np_activity
    complete_np_wifi_loc[:, 1] = 1
    complete_np_wifi_loc = np.array(complete_np_wifi_loc)
    np_wifi_loc = df_wifi_loc.values
    for i, date1 in enumerate(np_wifi_loc[:, 0]):
        for j, date2 in enumerate(complete_np_wifi_loc[:, 0]):
            if date1 == date2:
                complete_np_wifi_loc[j] = np_wifi_loc[i]
                break
    df_wifi_loc = pd.DataFrame(complete_np_wifi_loc, columns=['start_date', 'number of distinct locations'])
    df_wifi_loc = final_check(df_wifi_loc)
    df_wifi_loc.to_excel('./extracting/wifi_location/wifi_location_' + uid + '.xlsx', index=False)

def conversation_preprocess(uid):
    file = './dataset/sensing/conversation/conversation_' + uid + '.csv'
    df_conversation_raw = pd.read_csv(file)
    df_conversation_raw['start_date'] = df_conversation_raw['start_timestamp'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    df_conversation_raw['start_time'] = df_conversation_raw['start_timestamp'].apply(lambda x: time.strftime("%H:%M:%S", time.gmtime(x)))
    df_conversation_raw['end_date'] = df_conversation_raw[' end_timestamp'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    df_conversation_raw['end_time'] = df_conversation_raw[' end_timestamp'].apply(lambda x: time.strftime("%H:%M:%S", time.gmtime(x)))
    df_conversation = df_conversation_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    df_conversation['conversation_duration'] = duration_calc(df_conversation)
    df_conversation = df_conversation[['start_date', 'conversation_duration']]
    df_conversation = df_conversation.sort_values(by='start_date', ascending=True)
    df_conversation_date = df_conversation.groupby('start_date', as_index=False).mean()
    df_conversation_date.columns = ['start_date', 'activity inference']
    df_conversation_mean = df_conversation.groupby('start_date', as_index=False).mean()
    df_conversation_mean.columns = ['start_date', 'df_conversation_mean']
    df_conversation_skew = df_conversation.groupby('start_date', as_index=False).skew()
    df_conversation_skew.columns = ['start_date', 'df_conversation_skew']
    df_conversation_var = df_conversation.groupby('start_date', as_index=False).var(ddof=0)
    df_conversation_var.columns = ['start_date', 'df_conversation_var']
    df_conversation_std = df_conversation.groupby('start_date', as_index=False).std(ddof=0)
    df_conversation_std.columns = ['start_date', 'df_conversation_std']
    df_conversation_median = df_conversation.groupby('start_date', as_index=False).median()
    df_conversation_median.columns = ['start_date', 'df_conversation_median']
    df_conversation_sum = df_conversation.groupby('start_date', as_index=False).sum()
    df_conversation_sum.columns = ['start_date', 'df_conversation_sum']
    df_conversation_min = df_conversation.groupby('start_date', as_index=False).min()
    df_conversation_min.columns = ['start_date', 'df_conversation_min']
    df_conversation_max = df_conversation.groupby('start_date', as_index=False).max()
    df_conversation_max.columns = ['start_date', 'df_conversation_max']
    df_conversation = [df_conversation_date['start_date'],
                       df_conversation_mean['df_conversation_mean'],
                       df_conversation_skew['df_conversation_skew'],
                       df_conversation_var['df_conversation_var'],
                       df_conversation_std['df_conversation_std'],
                       df_conversation_median['df_conversation_median'],
                       df_conversation_sum['df_conversation_sum'],
                       df_conversation_min['df_conversation_min'],
                       df_conversation_max['df_conversation_max']]
    df_conversation = pd.concat(df_conversation, axis=1)
    df_conversation = final_check(df_conversation)
    df_conversation.to_csv('./extracting/conversation/conversation_' + uid + '.csv', index=False)

def dark_preprocess(uid):
    file = './dataset/sensing/dark/dark_' + uid + '.csv'
    df_dark_raw = pd.read_csv(file)
    df_dark_raw['start_date'] = df_dark_raw['start'].apply(lambda x: time.strftime("%Y-%m-%d %H", time.gmtime(x)))
    df_dark_raw['start_time'] = df_dark_raw['start'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))
    df_dark_raw['end_date'] = df_dark_raw['end'].apply(lambda x: time.strftime("%Y-%m-%d %H", time.gmtime(x)))
    df_dark_raw['end_time'] = df_dark_raw['end'].apply(lambda x: time.strftime("%M:%S", time.gmtime(x)))
    df_dark = df_dark_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    df_dark['dark_duration'] = duration_calc(df_dark)
    df_dark = df_dark[['start_date', 'dark_duration']]
    df_dark = df_dark.sort_values(by='start_date', ascending=True)
    df_dark_date = df_dark.groupby('start_date', as_index=False).mean()
    df_dark_date.columns = ['start_date', 'activity inference']
    df_dark_mean = df_dark.groupby('start_date', as_index=False).mean()
    df_dark_mean.columns = ['start_date', 'df_dark_mean']
    df_dark_skew = df_dark.groupby('start_date', as_index=False).skew()
    df_dark_skew.columns = ['start_date', 'df_dark_skew']
    df_dark_var = df_dark.groupby('start_date', as_index=False).var(ddof=0)
    df_dark_var.columns = ['start_date', 'df_dark_var']
    df_dark_std = df_dark.groupby('start_date', as_index=False).std(ddof=0)
    df_dark_std.columns = ['start_date', 'df_dark_std']
    df_dark_median = df_dark.groupby('start_date', as_index=False).median()
    df_dark_median.columns = ['start_date', 'df_dark_median']
    df_dark_sum = df_dark.groupby('start_date', as_index=False).sum()
    df_dark_sum.columns = ['start_date', 'df_dark_sum']
    df_dark_min = df_dark.groupby('start_date', as_index=False).min()
    df_dark_min.columns = ['start_date', 'df_dark_min']
    df_dark_max = df_dark.groupby('start_date', as_index=False).max()
    df_dark_max.columns = ['start_date', 'df_dark_max']
    df_dark = [df_dark_date['start_date'],
               df_dark_mean['df_dark_mean'],
               df_dark_skew['df_dark_skew'],
               df_dark_var['df_dark_var'],
               df_dark_std['df_dark_std'],
               df_dark_median['df_dark_median'],
               df_dark_sum['df_dark_sum'],
               df_dark_min['df_dark_min'],
               df_dark_max['df_dark_max']]
    df_dark = pd.concat(df_dark, axis=1)
    df_dark = final_check(df_dark)
    df_dark.to_csv('./extracting/dark/dark_' + uid + '.csv', index=False)

def gps_preprocess(uid):
    file = './dataset/sensing/gps/gps_' + uid + '.csv'
    df_gps_raw = pd.read_csv(file, index_col=False)
    df_gps_raw['date'] = df_gps_raw['time'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    df_gps = df_gps_raw[['date', 'latitude', 'longitude', 'travelstate']]
    df_gps['travelstate_one_hot'] = [0 if val == 'stationary' else 1 for val in df_gps['travelstate']]
    df_gps = df_gps.drop(['travelstate'], axis=1)
    df_gps = df_gps.sort_values(by='date', ascending=True)
    df_gps_date = df_gps.groupby('date', as_index=False).mean()
    df_gps_date.columns = ['date', 'latitude', 'longitude', 'travelstate']
    df_gps_mean = df_gps.groupby('date', as_index=False).mean()
    df_gps_mean.columns = ['date', 'latitude_mean', 'longitude_mean', 'travelstate_mean']
    df_gps_skew = df_gps.groupby('date', as_index=False).skew()
    df_gps_skew.columns = ['date', 'latitude_skew', 'longitude_skew', 'travelstate_skew']
    df_gps_var = df_gps.groupby('date', as_index=False).var(ddof=0)
    df_gps_var.columns = ['date', 'latitude_var', 'longitude_var', 'travelstate_var']
    df_gps_std = df_gps.groupby('date', as_index=False).std(ddof=0)
    df_gps_std.columns = ['date', 'latitude_std', 'longitude_std', 'travelstate_std']
    df_gps_median = df_gps.groupby('date', as_index=False).median()
    df_gps_median.columns = ['date', 'latitude_median', 'longitude_median', 'travelstate_median']
    df_gps_sum = df_gps.groupby('date', as_index=False).sum()
    df_gps_sum.columns = ['date', 'latitude_sum', 'longitude_sum', 'travelstate_sum']
    df_gps_min = df_gps.groupby('date', as_index=False).min()
    df_gps_min.columns = ['date', 'latitude_min', 'longitude_min', 'travelstate_min']
    df_gps_max = df_gps.groupby('date', as_index=False).max()
    df_gps_max.columns = ['date', 'latitude_max', 'longitude_max', 'travelstate_max']
    df_gps = [df_gps_date['date'],
              df_gps_mean['latitude_mean'], df_gps_mean['longitude_mean'], df_gps_mean['travelstate_mean'],
              df_gps_skew['latitude_skew'], df_gps_skew['longitude_skew'], df_gps_skew['travelstate_skew'],
              df_gps_var['latitude_var'], df_gps_var['longitude_var'], df_gps_var['travelstate_var'],
              df_gps_std['latitude_std'], df_gps_std['longitude_std'], df_gps_std['travelstate_std'],
              df_gps_median['latitude_median'], df_gps_median['longitude_median'], df_gps_median['travelstate_median'],
              df_gps_sum['latitude_sum'], df_gps_sum['longitude_sum'], df_gps_sum['travelstate_sum'],
              df_gps_min['latitude_min'], df_gps_min['longitude_min'], df_gps_min['travelstate_min'],
              df_gps_max['latitude_max'], df_gps_max['longitude_max'], df_gps_max['travelstate_max']]
    df_gps = pd.concat(df_gps, axis=1)
    df_gps = final_check(df_gps)
    df_gps.to_csv('./extracting/gps/gps_' + uid + '.csv', index=False)

def phone_charge_preprocess(uid):
    file = './dataset/sensing/phonecharge/phonecharge_' + uid + '.csv'
    df_phonecharge_raw = pd.read_csv(file)
    df_phonecharge_raw['start_date'] = df_phonecharge_raw['start'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    df_phonecharge_raw['start_time'] = df_phonecharge_raw['start'].apply(lambda x: time.strftime("%H:%M:%S", time.gmtime(x)))
    df_phonecharge_raw['end_date'] = df_phonecharge_raw['end'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    df_phonecharge_raw['end_time'] = df_phonecharge_raw['end'].apply(lambda x: time.strftime("%H:%M:%S", time.gmtime(x)))
    df_phonecharge = df_phonecharge_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    df_phonecharge['phonecharge_duration'] = duration_calc(df_phonecharge)
    df_phonecharge = df_phonecharge[['start_date', 'phonecharge_duration']]
    df_phonecharge = df_phonecharge.sort_values(by='start_date', ascending=True)
    df_phonecharge_date = df_phonecharge.groupby('start_date', as_index=False).mean()
    df_phonecharge_date.columns = ['start_date', 'activity inference']
    df_phonecharge_mean = df_phonecharge.groupby('start_date', as_index=False).mean()
    df_phonecharge_mean.columns = ['start_date', 'df_phonecharge_mean']
    df_phonecharge_skew = df_phonecharge.groupby('start_date', as_index=False).skew()
    df_phonecharge_skew.columns = ['start_date', 'df_phonecharge_skew']
    df_phonecharge_var = df_phonecharge.groupby('start_date', as_index=False).var(ddof=0)
    df_phonecharge_var.columns = ['start_date', 'df_phonecharge_var']
    df_phonecharge_std = df_phonecharge.groupby('start_date', as_index=False).std(ddof=0)
    df_phonecharge_std.columns = ['start_date', 'df_phonecharge_std']
    df_phonecharge_median = df_phonecharge.groupby('start_date', as_index=False).median()
    df_phonecharge_median.columns = ['start_date', 'df_phonecharge_median']
    df_phonecharge_sum = df_phonecharge.groupby('start_date', as_index=False).sum()
    df_phonecharge_sum.columns = ['start_date', 'df_phonecharge_sum']
    df_phonecharge_min = df_phonecharge.groupby('start_date', as_index=False).min()
    df_phonecharge_min.columns = ['start_date', 'df_phonecharge_min']
    df_phonecharge_max = df_phonecharge.groupby('start_date', as_index=False).max()
    df_phonecharge_max.columns = ['start_date', 'df_phonecharge_max']
    df_phonecharge = [df_phonecharge_date['start_date'],
                      df_phonecharge_mean['df_phonecharge_mean'],
                      df_phonecharge_skew['df_phonecharge_skew'],
                      df_phonecharge_var['df_phonecharge_var'],
                      df_phonecharge_std['df_phonecharge_std'],
                      df_phonecharge_median['df_phonecharge_median'],
                      df_phonecharge_sum['df_phonecharge_sum'],
                      df_phonecharge_min['df_phonecharge_min'],
                      df_phonecharge_max['df_phonecharge_max']]
    df_phonecharge = pd.concat(df_phonecharge, axis=1)
    df_phonecharge = final_check(df_phonecharge)
    df_phonecharge.to_csv('./extracting/phonecharge/phonecharge_' + uid + '.csv', index=False)

def phone_lock_preprocess(uid):
    file = './dataset/sensing/phonelock/phonelock_' + uid + '.csv'
    df_phonelock_raw = pd.read_csv(file)
    df_phonelock_raw['start_date'] = df_phonelock_raw['start'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    df_phonelock_raw['start_time'] = df_phonelock_raw['start'].apply(lambda x: time.strftime("%H:%M:%S", time.gmtime(x)))
    df_phonelock_raw['end_date'] = df_phonelock_raw['end'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    df_phonelock_raw['end_time'] = df_phonelock_raw['end'].apply(lambda x: time.strftime("%H:%M:%S", time.gmtime(x)))
    df_phonelock = df_phonelock_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    df_phonelock['phonelock_duration'] = duration_calc(df_phonelock)
    df_phonelock = df_phonelock[['start_date', 'phonelock_duration']]
    df_phonelock = df_phonelock.sort_values(by='start_date', ascending=True)
    df_phonelock_date = df_phonelock.groupby('start_date', as_index=False).mean()
    df_phonelock_date.columns = ['start_date', 'activity inference']
    df_phonelock_mean = df_phonelock.groupby('start_date', as_index=False).mean()
    df_phonelock_mean.columns = ['start_date', 'df_phonelock_mean']
    df_phonelock_skew = df_phonelock.groupby('start_date', as_index=False).skew()
    df_phonelock_skew.columns = ['start_date', 'df_phonelock_skew']
    df_phonelock_var = df_phonelock.groupby('start_date', as_index=False).var(ddof=0)
    df_phonelock_var.columns = ['start_date', 'df_phonelock_var']
    df_phonelock_std = df_phonelock.groupby('start_date', as_index=False).std(ddof=0)
    df_phonelock_std.columns = ['start_date', 'df_phonelock_std']
    df_phonelock_median = df_phonelock.groupby('start_date', as_index=False).median()
    df_phonelock_median.columns = ['start_date', 'df_phonelock_median']
    df_phonelock_sum = df_phonelock.groupby('start_date', as_index=False).sum()
    df_phonelock_sum.columns = ['start_date', 'df_phonelock_sum']
    df_phonelock_min = df_phonelock.groupby('start_date', as_index=False).min()
    df_phonelock_min.columns = ['start_date', 'df_phonelock_min']
    df_phonelock_max = df_phonelock.groupby('start_date', as_index=False).max()
    df_phonelock_max.columns = ['start_date', 'df_phonelock_max']
    df_phonelock = [df_phonelock_date['start_date'],
                    df_phonelock_mean['df_phonelock_mean'],
                    df_phonelock_skew['df_phonelock_skew'],
                    df_phonelock_var['df_phonelock_var'],
                    df_phonelock_std['df_phonelock_std'],
                    df_phonelock_median['df_phonelock_median'],
                    df_phonelock_sum['df_phonelock_sum'],
                    df_phonelock_min['df_phonelock_min'],
                    df_phonelock_max['df_phonelock_max']]
    df_phonelock = pd.concat(df_phonelock, axis=1)
    df_phonelock = final_check(df_phonelock)
    df_phonelock.to_csv('./extracting/phonelock/phonelock_' + uid + '.csv', index=False)

def sleep_preprocess(uid):
    file = './dataset/EMA/response/Sleep/Sleep_' + uid + '.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)
    sleep_ema_raw = pd.DataFrame.from_dict(dict_train)
    sleep_ema_clean = sleep_ema_raw.drop(['location'], axis=1)
    sleep_ema_clean['date'] = sleep_ema_clean['resp_time'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    sleep_ema = sleep_ema_clean[['date', 'hour', 'rate', 'social']]
    sleep_ema = sleep_ema.sort_values(by='date', ascending=True)
    sleep_ema_date = sleep_ema.groupby('date', as_index=False).mean()
    sleep_ema_date.columns = ['date', 'Mean_hour', 'Mean_rate', 'Mean_social']
    sleep_ema_mean = sleep_ema.groupby('date', as_index=False).mean()
    sleep_ema_mean.columns = ['date', 'Mean_hour', 'Mean_rate', 'Mean_social']
    sleep_ema_std = sleep_ema.groupby('date', as_index=False).std(ddof=0)
    sleep_ema_std.columns = ['date', 'Std_hour', 'Std_rate', 'Std_social']
    sleep_ema_median = sleep_ema.groupby('date', as_index=False).median()
    sleep_ema_median.columns = ['date', 'Median_hour', 'Median_rate', 'Median_social']
    sleep_ema_min = sleep_ema.groupby('date', as_index=False).min()
    sleep_ema_min.columns = ['date', 'Min_hour', 'Min_rate', 'Min_social']
    sleep_ema_max = sleep_ema.groupby('date', as_index=False).max()
    sleep_ema_max.columns = ['date', 'Max_hour', 'Max_rate', 'Max_social']
    sleep_ema_skew = sleep_ema.groupby('date', as_index=False).skew()
    sleep_ema_skew.columns = ['date', 'Skew_hour', 'Skew_rate', 'Skew_social']
    sleep_ema_var = sleep_ema.groupby('date', as_index=False).var(ddof=0)
    sleep_ema_var.columns = ['date', 'Var_hour', 'Var_rate', 'Var_social']
    sleep_ema_sum = sleep_ema.groupby('date', as_index=False).sum()
    sleep_ema_sum.columns = ['date', 'Sum_hour', 'Sum_rate', 'Sum_social']
    sleep_ema = [sleep_ema_date['date'],
                 sleep_ema_mean['Mean_hour'], sleep_ema_mean['Mean_rate'], sleep_ema_mean['Mean_social'],
                 sleep_ema_skew['Skew_hour'], sleep_ema_skew['Skew_rate'], sleep_ema_skew['Skew_social'],
                 sleep_ema_std['Std_hour'], sleep_ema_std['Std_rate'], sleep_ema_std['Std_social'],
                 sleep_ema_median['Median_hour'], sleep_ema_median['Median_rate'], sleep_ema_median['Median_social'],
                 sleep_ema_sum['Sum_hour'], sleep_ema_sum['Sum_rate'], sleep_ema_sum['Sum_social'],
                 sleep_ema_min['Min_hour'], sleep_ema_min['Min_rate'], sleep_ema_min['Min_social'],
                 sleep_ema_max['Max_hour'], sleep_ema_max['Max_rate'], sleep_ema_max['Max_social']]
    sleep_ema = pd.concat(sleep_ema, axis=1)
    sleep_ema = final_check(sleep_ema)
    sleep_ema.to_csv('./extracting/sleep/sleep_' + uid + '.csv', index=False)

def social_preprocess(uid):
    file = './dataset/EMA/response/Social/Social_' + uid + '.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)
    social_ema_clean = pd.DataFrame.from_dict(dict_train)
    social_ema_clean['date'] = social_ema_clean['resp_time'].apply(lambda x: time.strftime("%Y-%m-%d", time.gmtime(x)))
    social_ema = social_ema_clean[['date', 'number']]
    social_ema['number'].iloc[2] = 4
    social_ema['number'].iloc[3] = 3
    social_ema_date = social_ema.groupby('date', as_index=False).mean()
    social_ema_date.columns = ['date', 'Mean_number']
    social_ema_mean = social_ema.groupby('date', as_index=False).mean()
    social_ema_mean.columns = ['date', 'Mean_number']
    social_ema_std = social_ema.groupby('date', as_index=False).std(ddof=0)
    social_ema_std.columns = ['date', 'Std_number']
    social_ema_median = social_ema.groupby('date', as_index=False).median()
    social_ema_median.columns = ['date', 'Median_number']
    social_ema_min = social_ema.groupby('date', as_index=False).min()
    social_ema_min.columns = ['date', 'Min_number']
    social_ema_max = social_ema.groupby('date', as_index=False).max()
    social_ema_max.columns = ['date', 'Max_number']
    social_ema_skew = social_ema.groupby('date', as_index=False).skew()
    social_ema_skew.columns = ['date', 'Skew_number']
    social_ema_var = social_ema.groupby('date', as_index=False).var(ddof=0)
    social_ema_var.columns = ['date', 'Var_number']
    social_ema_sum = social_ema.groupby('date', as_index=False).sum()
    social_ema_sum.columns = ['date', 'Sum_number']
    social_ema = [social_ema_date['date'],
                  social_ema_mean["Mean_number"],
                  social_ema_std["Std_number"],
                  social_ema_median['Median_number'],
                  social_ema_min['Min_number'],
                  social_ema_max['Max_number'],
                  social_ema_skew['Skew_number'],
                  social_ema_sum['Sum_number']]
    social_ema = pd.concat(social_ema, axis=1)
    social_ema = final_check(social_ema)
    social_ema.to_csv('./extracting/social/social_' + uid + '.csv', index=False)

