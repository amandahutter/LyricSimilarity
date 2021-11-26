import torch
from torch.utils.data import Dataset
import re
import numpy as np
import sqlite3
from typing import List

class MusixMatchSqliteDataset(Dataset):
    def __init__(self, mxm_db_path: str, lastfm_db_path: str, filter_by_last_fm: bool, use_test: bool):
        use_test = int(use_test)
        con = sqlite3.connect(mxm_db_path)
        cur = con.cursor()
        if filter_by_last_fm:
            cur.execute('ATTACH DATABASE ? AS lastfm_sim', (lastfm_db_path,))

        self.__track_to_idx = {}
        self.__idx_to_track = {}

        if filter_by_last_fm:
            cur.execute("""
                SELECT DISTINCT track_id
                FROM lyrics
                WHERE is_test=?
                    AND track_id in
                        (SELECT tid FROM lastfm_sim.similars_src);
            """, (use_test,))
        else:
             cur.execute("""
                SELECT DISTINCT track_id
                FROM lyrics
                WHERE is_test=?
            """, (use_test,))

        track_id_tups = cur.fetchall()
        
        for i, track_id_tup in enumerate(track_id_tups):
            self.__track_to_idx[track_id_tup[0]] = i
            self.__idx_to_track[i] = track_id_tup[0]
        
        num_tracks = len(track_id_tups)

        word_tups = cur.execute('SELECT * FROM words;').fetchall()
        self.__word_idxs = {}
        for i, word_tup in enumerate(word_tups):
            self.__word_idxs[word_tup[0]] = i

        if filter_by_last_fm:
            cur.execute("""
                SELECT *
                FROM lyrics
                WHERE is_test=?
                    AND track_id in
                        (SELECT tid FROM lastfm_sim.similars_src);
            """, (use_test,))
        else:
            cur.execute("""
                SELECT *
                FROM lyrics
                WHERE is_test=?
            """, (use_test,))

        rows = cur.fetchall()

        self.data = np.zeros((num_tracks, len(self.__word_idxs.keys())))

        for i, row in enumerate(rows):
            if i % 1000 == 999:
                # Writes these logs on one line
                print('\r' + f'Loaded {i+1} word counts', end="")
            track_id, _, word, count, _ = row
            self.data[self.__track_to_idx[track_id]][self.__word_idxs[word]] = count
        print('\r' + f'Loaded {i+1} word counts')

        cur.close()

    def get_words(self)-> List[str]:
        return self.__word_idxs.keys()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (torch.Tensor(self.data[index]), self.__idx_to_track[index])

class MusixMatchCsvDataset(Dataset):
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

        


