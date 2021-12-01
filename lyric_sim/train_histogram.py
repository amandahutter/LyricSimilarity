import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datasets.mxm import MxMLastfmJoinedDataset
from utils import parse_args_and_config

from models.histogram import HistogramModel

config = parse_args_and_config()

N = config['batch_size']

# The batch size must be divisible by two, since we are splitting one batch using torch.split for src and dest nodes
assert N % 2 == 0

mxm_db = config['mxm_db']
hidden_size = config['hidden_size']
num_workers = config['num_workers']

print('Loading Musix Match training data...')
trainset = MxMLastfmJoinedDataset(mxm_db, False)
trainloader = DataLoader(trainset, N, shuffle=True, num_workers=num_workers)

NUM_WORDS = 5000

model = HistogramModel(NUM_WORDS*2, hidden_size)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(config['num_epochs']):
    print(f'Training epoch {epoch+1}')

    running_loss = 0.0
    for (i, batch) in enumerate(trainloader, 0):
        inputs, labels = batch

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        if i % 2000 == 0:
            print('[%d, %5d] loss: %.3f' % 
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

