from torch.utils.data import Dataset, DataLoader

with open('./files/mxm_dataset_train.txt') as train_file:
    training_set = Dataset()