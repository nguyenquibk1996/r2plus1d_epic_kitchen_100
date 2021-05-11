import pandas as pd
import random

# train file
path = '..\epic-kitchens-100\data\EPIC_100_train.csv'
df = pd.read_csv(path)
df = df.filter(items=['video_id', 'start_frame', 'stop_frame', 'narration'], axis='columns')

epic_100_video = []
video_list = list(set(df['video_id'].to_numpy()))

for video in video_list:
    if len(video) == 7:
        epic_100_video.append(video)
print(len(epic_100_video)) # 201
# shuffle list
random.shuffle(epic_100_video)
train_list = epic_100_video[:161]
valid_list = epic_100_video[161:]

temp_train_df = []
temp_valid_df = []

for video in train_list:
    row = df[df['video_id'] == video]
    temp_train_df.append(row)
    
train_df = pd.concat(temp_train_df)
train_df.to_csv('train.csv')


for video in valid_list:
    row = df[df['video_id'] == video]
    temp_valid_df.append(row)
    
valid_list = pd.concat(temp_valid_df)
valid_list.to_csv('valid.csv')