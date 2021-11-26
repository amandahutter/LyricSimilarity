import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datasets.mxm import MusixMatchDataset
from utils import parse_args_and_config
from adjacency_list.adjacency_list import AdjacencyList, SongNotFoundException
from models.histogram import HistogramModel

config = parse_args_and_config()

# The batch size must be divisible by two, since we are splitting one batch using torch.split for src and dest nodes
assert config['batch_size'] % 2 == 0

N = config['batch_size']

print('Loading Musix Match training data...')
trainset = MusixMatchDataset(config['mxm_train_file'])
trainloader = DataLoader(trainset, N, shuffle=True, num_workers=config['num_workers'])

print('Loading Musix Match testing data...')
testset = MusixMatchDataset(config['mxm_test_file'])
testloader = DataLoader(trainset, N, shuffle=True, num_workers=config['num_workers'])

print('Loading last fm adjacency list...')
adjacency_list = AdjacencyList(config['lastfm_db'])

num_words = len(trainset.get_words())

model = HistogramModel(num_words, 8192)

# TODO: update this with the custom loss module
criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# we will split the batch into two parts, once for srcs one for dests
N = int(N/2)

for epoch in range(config['num_epochs']):

    running_loss = 0.0
    for (i, data) in enumerate(trainloader, 0):
        inputs, labels = data

        try:
            srcInputs, destInputs = torch.split(inputs, N)
        except ValueError:
            # don't know how many epochs it will take to hit this but probably won't happen
            print('reached end of data, unable to split batch into equal parts')
            break
        srcLabels = labels[:N]
        destLabels = labels[N:]

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(srcInputs, destInputs)

        # get similarities
        targets = torch.zeros((N, 1))
        for i in range(N):
            try:
                targets[i] = adjacency_list.get_similarity(srcLabels[i], destLabels[i])
            except SongNotFoundException as ex:
                # not sure if this will actually happen during training. tested this script on a subset of data so this error was thrown.
                print(ex)
                targets[i] = 0

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

