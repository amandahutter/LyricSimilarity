import torch
from torch.utils.data import Dataset
import pandas as pd
import re
import numpy as np

class MusixMatchDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath) as inputfile:
            skiprows = None
            for i, line in enumerate(inputfile):
                if line[0] is '#':
                    continue
                elif line[0] is '%':
                    self.__words = self.__parse_word_list__(line)
                elif skiprows is None:
                    skiprows = i + 1
                    max = len(line.split(','))
                else:
                    length = len(line.split(','))
                    if length > max:
                        max = length
        names = []
        dtype = {}
        for i in range(max):
            names.append(i)
            dtype[i] = str
        self.data = pd.read_csv(filepath, skiprows=skiprows, names=names, dtype=dtype)

    def __len__(self):
        return len(self.data)

    def __get_item__(self, index):
        row = self.data.iloc[index, 1:]
        item = torch.zeros(len(self.__words))
        for token in row[2:]:
            if token is np.NAN:
                break
            split = re.sub(r"{}" ,"", str(token)).split(':')
            song_index, count = split
            item[int(song_index)] = int(count)
        return item

    @staticmethod
    def __parse_word_list__(line):
        return line[1:].split(',')

    def get_words(self):
        return self.__words

        


