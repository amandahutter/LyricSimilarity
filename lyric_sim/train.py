from torch.utils.data import DataLoader
from datasets.mxm import MusixMatchDataset
from utils import parse_args_and_config
from adjacency_list.adjacency_list import AdjacencyList
from models.histogram import HistogramModel

config = parse_args_and_config()

print('Loading Musix Match training data...')
trainset = MusixMatchDataset('./data_files/mxm_dataset_train.txt')
trainloader = DataLoader(trainset, config['batch_size'],
                                          shuffle=True, num_workers=2)

print('Loading Musix Match testing data...')
testset = MusixMatchDataset('./data_files/mxm_dataset_test.txt')
testloader = DataLoader(trainset, config['batch_size'],
                                          shuffle=True, num_workers=2)

print('Loading last fm adjacency list...')
adjacency_list = AdjacencyList()

model = HistogramModel()

for epoch in range(config['num_epochs']):

    running_loss = 0.0

    for (i, data) in enumerate(trainloader, 0):
        continue
