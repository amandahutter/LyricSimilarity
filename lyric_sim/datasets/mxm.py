import torch
from torch.utils.data import Dataset
import re
import numpy as np

class MusixMatchDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath) as inputfile:
            # find length of file for later
            num_rows = sum(1 for _ in inputfile)

            # reset the pointer
            inputfile.seek(0)
            for i, line in enumerate(inputfile):
                if line[0] is '#':
                    continue
                elif line[0] is '%':
                    self.__words = self.__parse_word_list__(line)
                    # we have reached the beginning of this section, find the length
                    # of the file and initialize the data matrix
                    skip_rows = i
                    self.data = np.zeros((num_rows-skip_rows-1, len(self.__words)))
                else:
                    for token in line.split(',')[2:]:
                        word_index, count = re.sub(r"{}" ,"", str(token)).split(':')

                        # the song indices in the file are 1-indexed
                        self.data[i-skip_rows-1, int(word_index)-1] = int(count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.data[index])

    @staticmethod
    def __parse_word_list__(line):
        return line[1:].split(',')

    def get_words(self):
        return self.__words

        


