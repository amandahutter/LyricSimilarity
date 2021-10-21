from torch.utils.data import Dataset
import pandas as pd

class MusixMatchDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath) as inputfile:
            for i, line in enumerate(inputfile):
                if line[0] is '#':
                    continue
                elif line[0] is '%':
                    self.__classes = self.__parse_word_list__(line)
                else:
                    skip_rows = i + 1
                    break
        self.data = pd.read_csv(filepath, skip_rows=skip_rows)

    def __len__(self):
        return len(self.data)

    def __get_item__(self, index):
        line = self.data.iloc[index, 1:]
        item = torch.zeros(len(self.__classes))
        for token in line.split(','):
            song_index, count = token.split(':')
            item[song_index] = count
        return item

    @staticmethod
    def __parse_word_list__(line):
        return line[1:].split(',')

    def get_classes():
        return self.__classes

        


