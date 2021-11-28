import os
import torch
from torch.utils.data import Dataset
import re
import numpy as np
import sqlite3
from typing import List

NUM_WORDS = 5000

class MxMLastfmJoinedDataset(Dataset):
    """
    Constructs a series of examples [1x10000] where the first 5000 words are the word counts for song 1, and the second 5000 are word counds for song two.
    The labels are similarity score threshold between the two songs is_similar(s1, s2) >= .5 ? 1 : 0. Uses the similars_src table from lastfm only.
    """

    def __init__(self, mxm_db_path: str, lastfm_db_path: str, use_test: bool, use_saved_data: bool):

        use_test = int(use_test)

        con = sqlite3.connect(mxm_db_path)
        cur = con.cursor()
        cur.execute('ATTACH DATABASE ? AS lastfm_sim', (lastfm_db_path,))

        # TODO: add logic for train and test
        if use_saved_data and os.path.exists('./data_files/histogram_examples.npy') and os.path.exists('./data_files/histogram_labels.npy'):
            self.__data = np.load('./data_files/histogram_examples.npy')
            self.__labels = np.load('./data_files/histogram_labels.npy')
            return

        con = sqlite3.connect(mxm_db_path)
        cur = con.cursor()
        cur.execute('ATTACH DATABASE ? AS lastfm_sim', (lastfm_db_path,))

        word_tups = cur.execute('SELECT * FROM words;').fetchall()
        self.__word_idxs = {}
        for i, word_tup in enumerate(word_tups):
            self.__word_idxs[word_tup[0]] = i

        print('Fetching data...')

        self.__data = np.empty([])
        self.__labels = np.empty([])

        # create each example one by one
        track_ids = cur.execute("""
            SELECT DISTINCT(lyrics.track_id)
            FROM lyrics
            INNER JOIN lastfm_sim.similars_src
            ON lyrics.track_id = lastfm_sim.similars_src.tid
            WHERE lyrics.is_test = ?
        """, (use_test,)).fetchall()

        for i, track_id_tup in enumerate(track_ids):
            track_id = track_id_tup[0]
            print(f'Creating example for track {track_id}.', end="")
            words = cur.execute("""
                SELECT lyrics.word, lyrics.count
                FROM lyrics
                WHERE lyrics.is_test = ? AND
                lyrics.track_id = ?;
            """, (use_test, str(track_id),)).fetchall()
            
            example = np.zeros((NUM_WORDS*2))
            for word, count in words:
                example[self.__word_idxs[word]] = count

            similars = cur.execute("""
                SELECT lastfm_sim.similars_src.target
                FROM lastfm_sim.similars_src
                WHERE tid = ?
            """, (track_id,)).fetchone()

            similars_tokens = similars[0].split(',')

            example_copy = np.copy(example)

            print(f'Getting {len(similars_tokens)/2} similars for track {track_id}')
            for i in range(0, len(similars_tokens), 2):
                similar_id = similars_tokens[i]
                similar_score = 1 if float(similars_tokens[i+1]) > .5 else 0
                similar_words = cur.execute("""
                    SELECT lyrics.word, lyrics.count
                    FROM lyrics
                    WHERE lyrics.track_id = ?;
                """, (similar_id,)).fetchall()

                for similar in similar_words:
                    word, count = similar
                    example[self.__word_idxs[word]+NUM_WORDS] = count
                
                self.__data = np.append(self.__data, example_copy)
                self.__labels = np.append(self.__labels, similar_score)

                np.save('./data_files/histogram_examples.npy', self.__data)
                np.save('./data_files/histogram_labels.npy', self.__labels)

        def __len__(self):
            return len(self.__data)

        def __getitem__(self, index):
            return (torch.Tensor(self.__data[index]), self.__labels[index])
                    

class MusixMatchSqliteDataset(Dataset):
    """
    Deprecated. Filtered the musix match dataset depending on whether the id could be found in the last fm dataset.
    """
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

        


