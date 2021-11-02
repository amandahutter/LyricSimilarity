from torch.utils.data import DataLoader
from datasets.mxm import MusixMatchDataset
from utils import parse_args_and_config

config = parse_args_and_config()

print('Loading data...')
train_dataset = MusixMatchDataset('./data_files/mxm_dataset_train.txt')
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

print('First batch:')
example = next(iter(train_dataloader))
print(example)