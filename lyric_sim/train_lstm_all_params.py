import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datasets.lyrics import LyricsSqliteDataset
from utils import parse_args_and_config

from models.lstm import LSTM, CombinationType

config = parse_args_and_config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

lyrics_db_path = config['lyrics_db_path']
mxm_db_path = config['mxm_db_path']
lastfm_db_path = config['lastfm_db_path']
N = config['batch_size']
emb_size = config['emb_size']
hidden_size = config['hidden_size']
num_layers = config['num_layers']
dropout = config['dropout']
num_fc = config['num_fc']
combo_unit = CombinationType[config['combo_unit']]
learning_rate = config['learning_rate']
momentum = config['momentum']
num_workers = config['num_workers']
dropout_first = config['dropout_first']

print('Loading Lyrics training data...')
trainset = LyricsSqliteDataset(lyrics_db_path, mxm_db_path, lastfm_db_path, False, True)
print('Size of trainset: ', len(trainset))
trainloader = DataLoader(trainset, N, shuffle=True, num_workers=num_workers)

MODEL_PATH = f'./saved_models/{config["model_name"]}.pth'

input_size = len(trainset.vocab)

model = LSTM(input_size, emb_size, hidden_size, num_layers, dropout, num_fc, combo_unit, dropout_first)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

def save_model(model):
    print(f'Saving model to {MODEL_PATH}')
    torch.save(model.state_dict(), MODEL_PATH)
    print('Finished Training')

for epoch in range(config['num_epochs']):
    print(f'Training epoch {epoch+1}')

    running_loss = 0.0
    running_count = 0
    for (i, batch) in enumerate(trainloader, 0):
        inputs, labels = (batch[0][0].to(device), batch[0][1].to(device)), batch[1].to(device)
        
        #_, counts = labels.unique(sorted = True, return_counts = True)
        #print('Percent positive class in batch: ', counts[1] / counts.sum())

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs[0], inputs[1])

        #loss = criterion(outputs.squeeze(), labels)
        outputs = outputs.squeeze()
        outputs = outputs.to(torch.float)
        labels = labels.to(torch.long)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_count += 1

        if i % 1000 == 0: 
            print('[%d, %d] loss: %.3f' % 
                  (epoch, running_count, running_loss / running_count))

print('Finished Training')

save_model(model)

