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

print('Loading Musix Match testing data...')
testset = MxMLastfmJoinedDataset(mxm_db, True, num_examples=num_examples)
sampler = WeightedRandomSampler(testset.get_sample_weights(), testset.__len__())
testloader = DataLoader(testset, N, num_workers=num_workers, sampler=sampler)

NUM_WORDS = 5000

MODEL_PATH = f'./saved_models/{config["config_name"]}.pth'

model = HistogramModel(NUM_WORDS*2, hidden_size)
model.load_state_dict(torch.load(MODEL_PATH))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        input, labels = data
        outputs = model(input)

        outputs = outputs.squeeze()
        n = labels.size(0)
        total += n
        # Round the outputs to 0 or 1.
        for i in range(n):
            correct += int(torch.round(outputs[i]) == labels[i])

print(f'Accuracy of the network on {total} test examples: {100 * correct / total}%')