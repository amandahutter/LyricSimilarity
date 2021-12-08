import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
from datasets.lyrics import LyricsSqliteDataset
from utils import parse_args_and_config

from models.lstm import LSTM, CombinationType

config = parse_args_and_config()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device {device}')

N = config['batch_size']
lyrics_db_path = config['lyrics_db_path']
mxm_db_path = config['mxm_db_path']
lastfm_db_path = config['lastfm_db_path']
emb_size = config['emb_size']
hidden_size = config['hidden_size']
dropout = config['dropout']
num_fc = config['num_fc']
combo_unit = CombinationType(config['combo_unit'])
num_workers = config['num_workers']


print('Loading Lyrics testing data...')
testset = LyricsSqliteDataset(lyrics_db_path, mxm_db_path, lastfm_db_path, False, True)
testloader = DataLoader(testset, N, num_workers=num_workers)

# Moved from above
input_size = len(testset.vocab)

MODEL_PATH = f'./saved_models/{config["model_name"]}.pth'

model = LSTM(input_size, emb_size, hidden_size, dropout, num_fc, combo_unit)
model.to(device)
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
        
        outputs = model(input[0], input[1])
        
        outputs =  torch.argmax(outputs, dim = 2)

        outputs = outputs.squeeze() 

        n = labels.size(0)
        total += n
        for i in range(n):

            label_pred = outputs[i]
            label = labels[i]

            TP += (label_pred == 1) & (label == 1)
            TN += (label_pred == 0) & (label == 0)
            FP += (label_pred == 1) & (label == 0)
            FN += (label_pred == 0) & (label == 1)

correct = TP + TN
print(f'Accuracy of the network on {total} test examples: {100 * correct / total}%')
print(f'F1-Score of the network on {total} test examples: {100 * (2*TP) / (2*TP + FP + FN)}%')

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
print(f'Labeled Positive Portion on {total} test examples: {100 * (TP + FN)/(total)}%')
print(f'Labeled Negative Portion on {total} test examples: {100 * (FP + TN)/(total)}%')

