from torch.utils.data import DataLoader
import torch.optim as optim
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

# TODO: update this with the custom loss module
# criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(config['num_epochs']):

    running_loss = 0.0
    for (i, data) in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

