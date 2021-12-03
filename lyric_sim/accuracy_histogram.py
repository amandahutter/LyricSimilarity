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
TP = 0
FP = 0 
TN = 0 
FN = 0 
other = 0 
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

            if ((int(torch.round(outputs[i])) == 1)  & (int(torch.round(labels[i])) == 1)):
                TP += 1 
            elif ((int(torch.round(outputs[i])) == 0)  & (int(torch.round(labels[i])) == 0)):
                TN += 1 
            elif ((int(torch.round(outputs[i])) == 1)  & (int(torch.round(labels[i])) == 0)):
                FP += 1
            elif ((int(torch.round(outputs[i])) == 0)  & (int(torch.round(labels[i])) == 1)):
                FN += 1 
            else: 
                other += 1 

print(f'Accuracy of the network on {total} test examples: {100 * correct / total}%')

print(f'Precision (Positive Preditive Value) on {total} test examples: {100 * TP/(TP + FP)}%')
print(f'Recall (True Positive Rate) on {total} test examples: {100 * TP/(TP + FN)}%')
print(f'Negative Preditive Value on {total} test examples: {100 * TN/(TN + FN)}%')
print(f'Specificity (True Negative Rate) on {total} test examples: {100 * TN/(TN + FP)}%')

print(f'True Positive Amount on {total} test examples: {100 * TP/(total)}%')
print(f'False Positive Amount on {total} test examples: {100 * FP/(total)}%')
print(f'True Negative Amount on {total} test examples: {100 * TN/(total)}%')
print(f'False Negative Amount on {total} test examples: {100 * FN/(total)}%')

print(f'Predicted Positive Portion on {total} test examples: {100 * (FP + TP)/(total)}%')
print(f'Predicted Negative Portion on {total} test examples: {100 * (FN + TN)/(total)}%')

