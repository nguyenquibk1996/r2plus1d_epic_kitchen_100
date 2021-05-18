import pandas as pd
import os


# train file
path = '..\epic-kitchens-100\data\EPIC_100_train.csv'
df = pd.read_csv(path)
df = df.filter(items=['video_id', 'start_frame', 'stop_frame', 'verb', 'verb_class', 'noun', 'noun_class', 'narration'], axis='columns')

epic_100_video = []
video_list = list(set(df['video_id'].to_numpy()))

train_video_list = os.listdir('..\epic-kitchens-100\data_train')
valid_video_list = os.listdir('..\epic-kitchens-100\data_valid')
print('train: ', len(train_video_list))
print('valid: ', len(valid_video_list))

temp_train_df = []
temp_valid_df = []

for video in train_video_list:
    row = df[df['video_id'] == video]
    temp_train_df.append(row)
train_df = pd.concat(temp_train_df)
train_df.to_csv('epic_100_train.csv')


for video in valid_video_list:
    row = df[df['video_id'] == video]
    temp_valid_df.append(row)
valid_list = pd.concat(temp_valid_df)
valid_list.to_csv('epic_100_valid.csv')