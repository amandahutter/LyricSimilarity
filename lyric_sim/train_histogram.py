import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
from datasets.mxm import MxMLastfmJoinedDataset
from utils import parse_args_and_config

from models.histogram import HistogramModel

config = parse_args_and_config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

N = config['batch_size']

mxm_db = config['mxm_db']
hidden_size = config['hidden_size']
num_workers = config['num_workers']
num_examples = config['num_examples']

print('Loading Musix Match training data...')
trainset = MxMLastfmJoinedDataset(mxm_db, False, num_examples=num_examples)
sampler = WeightedRandomSampler(trainset.get_sample_weights(), trainset.__len__())
trainloader = DataLoader(trainset, N, num_workers=num_workers, sampler=sampler)

NUM_WORDS = 5000

model = HistogramModel(NUM_WORDS*2, hidden_size)
model.to(device)

criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(config['num_epochs']):
    print(f'Training epoch {epoch+1}')

    running_loss = 0.0
    for (i, batch) in enumerate(trainloader, 0):
        inputs, labels = batch[0].to(device), batch[1].to(device)

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
                  (epoch, i, running_loss / 2000))
            running_loss = 0.0

MODEL_PATH = f'./data_files/{config["config_name"]}.pth'

print(f'Saving model to {MODEL_PATH}')
torch.save(model.state_dict(), MODEL_PATH)
print('Finished Training')

