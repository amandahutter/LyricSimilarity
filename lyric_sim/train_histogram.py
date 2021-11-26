import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datasets.mxm import MusixMatchSqliteDataset
from utils import parse_args_and_config
from adjacency_list.adjacency_list import AdjacencyList, SongNotFoundException
from models.histogram import HistogramModel

config = parse_args_and_config()

# The batch size must be divisible by two, since we are splitting one batch using torch.split for src and dest nodes
assert config['batch_size'] % 2 == 0

N = config['batch_size']
mxm_db = config['mxm_db']
lastfm_db = config['lastfm_db']
hidden_size = config['hidden_size']
filter_tracks = config['filter_tracks']
num_workers = config['num_workers']

print('Loading Musix Match training data...')
trainset = MusixMatchSqliteDataset(mxm_db, lastfm_db, filter_tracks, False)
trainloader = DataLoader(trainset, N, shuffle=True, num_workers=num_workers, drop_last=True)

print('Loading Musix Match testing data...')
testset = MusixMatchSqliteDataset(mxm_db, lastfm_db, filter_tracks, True)
testloader = DataLoader(trainset, N, shuffle=True, num_workers=num_workers, drop_last=True)

print('Loading last fm adjacency list...')
adjacency_list = AdjacencyList(lastfm_db, mxm_db, filter_tracks)

num_words = len(trainset.get_words())

model = HistogramModel(num_words, hidden_size)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# we will split the batch into two parts, once for srcs one for dests
N = int(N/2)

for epoch in range(config['num_epochs']):
    print(f'Training epoch {epoch+1}')

    running_loss = 0.0
    for (i, data) in enumerate(trainloader, 0):
        inputs, labels = data

        srcInputs, destInputs = torch.split(inputs, N)

        srcLabels = labels[:N]
        destLabels = labels[N:]

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(srcInputs, destInputs)

        # get similarities
        targets = torch.zeros((N, 1))
        for i in range(N):
            targets[i] = adjacency_list.get_similarity(srcLabels[i], destLabels[i])

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

