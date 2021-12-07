import torch
from torch.utils.data import Dataset
import re
import numpy as np
import sqlite3

NUM_WORDS = 5000

class MxMLastfmJoinedDataset(Dataset):
    """
    Constructs a series of examples [1x10000] where the first 5000 words are the word counts for song 1, and the second 5000 are word counds for song two.
    The labels are similarity score threshold between the two songs is_similar(s1, s2) >= .5 ? 1 : 0. Uses the similars_src table from lastfm only.
    """

    def __init__(self, mxm_lasfm_db_path, use_test: bool, num_examples=500000, similarity_threshold=0.5):

        use_test = int(use_test)

        con = sqlite3.connect(mxm_lasfm_db_path)
        cur = con.cursor()

        print('Fetching data...')
        examples = cur.execute("""
            SELECT similars_src.score, src.histogram, dest.histogram, src.track_id, dest.track_id
            FROM examples src, examples dest
            JOIN similars_src
            ON src.track_id = similars_src.src
            AND dest.track_id = similars_src.dest
            WHERE similars_src.is_test = ?
            AND dest.histogram != ''
            LIMIT ?;
        """, (use_test, num_examples,)).fetchall()

        self.__data = np.zeros((len(examples), 10000), dtype=np.int8)
        self.__labels = np.empty(len(examples), dtype=np.float32)

        # figure out class counts so we can sample later
        similar_count = 0

        print('Loading data into numpy array...')
        for i, example in enumerate(examples):
            if i % 10000 == 0:
                print('\r' + f'Loaded {i} word counts', end="")

            # 1 if simialar 0 otherwise
            is_similar = int(example[0] >= similarity_threshold)
            self.__labels[i] = is_similar
            similar_count += is_similar

            src_counts = example[1].split(',')
            for src_count in src_counts:
                word_idx, count = src_count.split(':')
                self.__data[i][int(word_idx)] = count

            dest_counts = example[2].split(',')
            for dest_count in dest_counts:
                word_idx, count = dest_count.split(':')
                self.__data[i][int(word_idx)+5000] = count

        print('\r' + f'Loaded {len(examples)} word counts',)

        #weights = np.array([0.1, 1])
        weights = np.array([similar_count/num_examples, (num_examples-similar_count)/num_examples])

        print(f'Will sample class <not similar> with {weights[0]} probability and <similar> with {weights[1]} probability')

        self.__sample_weights = np.array([weights[int(t)] for t in self.__labels])

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, index):
        return (torch.Tensor(self.__data[index]), self.__labels[index])

    def get_sample_weights(self):
        return self.__sample_weights

class MusixMatchCsvDataset(Dataset):
    """
    Exposes the musix match word count data as a list of tensors 1x5000 where the 5000 columns are each attributed to a word in the words list.
    e.g. [4, 2, 0, 0, 3, 1, ... , 0, 0, 1, 0, 0]
    """
    def __init__(self, filepath):
        self.__idx_to_mxmid = {}
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
                    tokens = line.split(',')
                    self.__idx_to_mxmid[i-skip_rows-1] = tokens[0]
                    for token in tokens[2:]:
                        word_index, count = re.sub(r"{}" ,"", str(token)).split(':')

                        # the word indices in the file are 1-indexed
                        self.data[i-skip_rows-1, int(word_index)-1] = int(count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (torch.Tensor(self.data[index]), self.__idx_to_mxmid[index])

    @staticmethod
    def __parse_word_list__(line):
        return line[1:].split(',')

    def get_words(self):
        return self.__words

        


