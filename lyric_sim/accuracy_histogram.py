import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
from datasets.mxm import MxMLastfmJoinedDataset
from utils import parse_args_and_config, write_results_to_csv

from models.histogram import HistogramModel

config = parse_args_and_config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

N = config['batch_size']

mxm_db = config['mxm_db']
learning_rate = config['learning_rate']
hidden_size_0 = config['hidden_size_0']
hidden_size_1 = config['hidden_size_1']
num_workers = config['num_workers']
num_examples = config['num_examples']
keep_words = config['keep_words']
config_name = config["config_name"]

print('Loading Musix Match testing data...')
testset = MxMLastfmJoinedDataset(mxm_db, True, num_examples=num_examples, keep_words=keep_words)
sampler = WeightedRandomSampler(testset.get_sample_weights(), testset.__len__())
testloader = DataLoader(testset, N, num_workers=num_workers, sampler=sampler)

MODEL_PATH = f'./saved_models/{config_name}.pth'

model = HistogramModel(keep_words*2, hidden_size_0, hidden_size_1)
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
        # Round the outputs to -1 or 1.
        for i in range(n):
            label_pred = 1 if outputs[i] > 0 else -1
            label = labels[i]

            TP += (label_pred == 1) & (label == 1)
            TN += (label_pred == -1) & (label == -1)
            FP += (label_pred == 1) & (label == -1)
            FN += (label_pred == -1) & (label == 1)

write_results_to_csv(TP.item(), TN.item(), FP.item(), FN.item(), f'./plots/{config_name}.csv')

correct = TP + TN
print(f'Accuracy of the network on {total} test examples: {100 * correct / total}%')
print(f'F1-Score of the network on {total} test examples: {100 * (2*TP) / (2*TP + FP + FN)}%')

print(f'Precision (Positive Predictive Value) on {total} test examples: {100 * TP/(TP + FP)}%')
print(f'Recall (True Positive Rate) on {total} test examples: {100 * TP/(TP + FN)}%')
print(f'Negative Preditive Value on {total} test examples: {100 * TN/(TN + FN)}%')
print(f'Specificity (True Negative Rate) on {total} test examples: {100 * TN/(TN + FP)}%')

print(f'True Positive Amount on {total} test examples: {100 * TP/(total)}%')
print(f'False Positive Amount on {total} test examples: {100 * FP/(total)}%')
print(f'True Negative Amount on {total} test examples: {100 * TN/(total)}%')
print(f'False Negative Amount on {total} test examples: {100 * FN/(total)}%')

print(f'Predicted Positive Portion on {total} test examples: {100 * (FP + TP)/(total)}%')
print(f'Predicted Negative Portion on {total} test examples: {100 * (FN + TN)/(total)}%')
print(f'Labeled Positive Portion on {total} test examples: {100 * (TP + FN)/(total)}%')
print(f'Labeled Negative Portion on {total} test examples: {100 * (FP + TN)/(total)}%')

