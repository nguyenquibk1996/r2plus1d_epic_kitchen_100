import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os
from random import randint
import time

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelBinarizer
import utils


# check cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}')

train_df = pd.read_csv('epic_100_train.csv')
valid_df = pd.read_csv('epic_100_valid.csv')

train_video_path = '..\epic-kitchens-100\data_train'
train_video_list = os.listdir(train_video_path)
valid_video_path = '..\epic-kitchens-100\data_valid'
valid_video_list = os.listdir(valid_video_path)

def stack_clip(df, n_frames, set_type):
    start_frame = df['start_frame'].tolist()
    stop_frame = df['stop_frame'].tolist()
    videos = df['video_id'].tolist()
    label = df['verb_class'].tolist()
    epic_100_data = {
        'video': [],
        'clip': [],
        'label': []
    }
    for i, video in tqdm(enumerate(videos)):
        if start_frame[i] < stop_frame[i]-int(n_frames):
            if set_type == 'train':
                path = os.path.join(train_video_path, video)
                epic_100_data['video'].append(path)
                frames = os.listdir(path)
            else:
                path = os.path.join(valid_video_path, video)
                epic_100_data['video'].append(path)
                frames = os.listdir(path)
            epic_100_data['label'].append(label[i])
            temporal_index = randint(start_frame[i], stop_frame[i]-int(n_frames))
            clips = frames[temporal_index:temporal_index+int(n_frames)]
            new_clip = []
            for clip in clips:
                clip = os.path.join(path, clip)
                new_clip.append(clip)
            epic_100_data['clip'].append(new_clip)
    return epic_100_data

epic_valid = stack_clip(valid_df, 16, 'valid')
# save file
# df_valid_to_file = pd.DataFrame(epic_valid, columns=['clip', 'label'])
# df_valid_to_file.to_csv('valid_data_input.csv', index=False)

epic_train = stack_clip(train_df, 16, 'train')
# df_train_to_file = pd.DataFrame(epic_train, columns=['clip', 'label'])
# df_train_to_file.to_csv('train_data_input.csv', index=False)

# custom dataset
class EPIC(Dataset):
    def __init__(self, clips, labels):
        self.clips = clips
        self.labels = labels

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]
        input_frames = []
        for frame in clip:
            image = Image.open(frame)
            image = image.convert('RGB')
            image = np.array(image)
            image = utils.transforms(image=image)['image']
            input_frames.append(image)
        input_frames = np.array(input_frames)
        input_frames = np.transpose(input_frames, (3,0,1,2))
        input_frames = torch.tensor(input_frames, dtype=torch.float32)
        input_frames = input_frames.to(device)
        # label
        self.labels = np.array(self.labels)
        lb = LabelBinarizer()
        self.labels = lb.fit_transform(self.labels)
        label = self.labels[i]
        label = torch.tensor(label, dtype=torch.long)
        label = label.to(device)
        return (input_frames, label)

X_train = epic_train['clip']
y_train = epic_train['label']
X_test = epic_valid['clip']
y_test = epic_valid['label']

train_data = EPIC(X_train, y_train)
val_data = EPIC(X_test, y_test)

# learning params
lr = 0.001
batch_size = 16

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# model
model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 87)
model = model.to(device)
print(model)

# criterion
criterion = nn.CrossEntropyLoss()

# optim
optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)

# scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=True
)

# training
def fit(model, train_dataloader):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(train_dataloader), total=int(len(train_data)/train_dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.max(target, 1)[1])
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        loss.backward()
        optimizer.step()
    
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')

    return train_loss, train_accuracy

# validation
def validate(model, test_dataloader):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=int(len(val_data)/val_loader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, torch.max(target, 1)[1])

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == torch.max(target, 1)[1]).sum().item()

        val_loss = val_running_loss/len(val_loader.dataset)
        val_accuracy = 100. * val_running_correct/len(val_loader.dataset)
        print(f'Val loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')

        return val_loss, val_accuracy


train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
epochs = 20
min_val_loss = np.Inf
epochs_no_impove = 0
n_epochs_stop = 6
early_stop = False
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_accuracy = fit(model, train_loader)
    val_epoch_loss, val_epoch_accuracy = validate(model, val_loader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    if val_epoch_loss < min_val_loss:
        epochs_no_impove = 0
        min_val_loss = val_epoch_loss
    else:
        epochs_no_impove += 1
        print('epochs_no_impove: ', epochs_no_impove)
    if epoch > 5 and epochs_no_impove == n_epochs_stop:
        print('Early stoppping!')
        early_stop = True
        break
    else:
        continue

    scheduler.step(val_epoch_loss)
    save_path = 'models/r2plus1d_{}.pth'.format(epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }, save_path)

end = time.time()

print(f'{(end-start)/60:.3f} minutes')

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/accuracy.png')
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/loss.png')
 
print('TRAINING COMPLETE')