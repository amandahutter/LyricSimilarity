import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re
import numpy as np
import pandas as pd
import sqlite3

MAX_LENGTH = 100
SIM_THRESHOLD = 0.5

class LyricsSqliteDataset(Dataset):
    def __init__(self, lyrics_db_path: str, mxm_db_path: str, lastfm_db_path: str, use_test: bool, pad_length: bool):
        use_test = int(use_test)
        
        # establish connection to lyrics
        con = sqlite3.connect(lyrics_db_path)
        cur = con.cursor()
        cur.execute('ATTACH DATABASE ? AS lastfm', (lastfm_db_path,))
        cur.execute('ATTACH DATABASE ? AS mxm', (mxm_db_path,))

        # read in lyrics
        qry = '''SELECT B.tid, A.mxm_id, A.lyrics_id, A.lyrics, A.explicit, B.is_test
                 FROM lyrics A 
                 JOIN (SELECT DISTINCT track_id AS tid, mxm_tid AS mxm_id, is_test FROM mxm.lyrics) B 
                 ON A.mxm_id = B.mxm_id
                 WHERE lyrics <> "" AND B.is_test = {}'''.format(use_test)
        df = pd.read_sql_query(qry, con)

        self.__idx_to_track = df[['tid']].to_dict(orient='dict')['tid']
             
        # clean data
        df['lyrics'] = df['lyrics'].str.replace('[^a-zA-Z]', ' ')
        df['lyrics'] = df['lyrics'].str.lower()
        df['lyrics'] = df['lyrics'].str.strip()
        df = df[df['lyrics'] != '']
        df.reset_index(inplace=True, drop=True)
        
        # create ids df for use later
        df_ids = df[['tid']].copy()
        df_ids.reset_index(inplace=True)
        
        # pad if requested (inc. cutting to MAX_LENGTH)
        if pad_length:
            df['lyrics'] = df['lyrics'].apply(lambda x: ' '.join(x.split()[:MAX_LENGTH] + ['<pad>']*(MAX_LENGTH - len(x.split()))))
        
        # define vocab & create data (tensor of ints for lyrics)
        tokenizer = get_tokenizer('basic_english')
        if use_test:
            self.vocab = torch.load('./data_files/lyrics_vocab.pth')
        else:
            vocab = build_vocab_from_iterator(map(tokenizer, df['lyrics']), specials=['<pad>'])
            vocab.set_default_index(0)
            self.vocab = vocab
            torch.save(vocab, './data_files/lyrics_vocab.pth')

        self.data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in df['lyrics']]
        
        # similarity score (from lastfm)
        qry = '''SELECT A.*
                 FROM lastfm.similars_src A
                 JOIN (SELECT DISTINCT track_id AS tid, mxm_tid AS mxm_id FROM mxm.lyrics) B ON A.src = B.tid
                 JOIN (SELECT DISTINCT track_id AS tid, mxm_tid AS mxm_id FROM mxm.lyrics) BB ON A.dest = BB.tid
                 JOIN lyrics C ON B.mxm_id = C.mxm_id
                 JOIN lyrics CC ON BB.mxm_id = CC.mxm_id'''
        df_sim_all = pd.read_sql_query(qry, con)
        
        # merge in ids to similarity data (will cut out remaining records we won't use)
        df_sim_all = df_sim_all.merge(df_ids, how='left', left_on='src', right_on='tid')
        df_sim_all = df_sim_all.rename(columns={'index': 'src_id'})
        df_sim_all = df_sim_all.merge(df_ids, how='left', left_on='dest', right_on='tid')
        df_sim_all = df_sim_all.rename(columns={'index': 'dest_id'})
        
        # keep if both tracks have lyrics
        df_sim_lyrics = df_sim_all[(~df_sim_all['src_id'].isnull()) & (~df_sim_all['dest_id'].isnull())].copy()
        
        # set 1/0 for similarity
        df_sim_lyrics['similar'] = np.where(df_sim_lyrics['score'] >= SIM_THRESHOLD, 1, 0)
        similar_count = df_sim_lyrics[df_sim_lyrics['similar'] == 1].shape[0]
        
        # keep all similars
        df_sim_yes = df_sim_lyrics[df_sim_lyrics['similar'] == 1].copy()

        # select random non-similar in equal number for 0s
        df_sim_no = df_sim_lyrics[df_sim_lyrics['similar'] == 0].sample(n=df_sim_yes.shape[0], replace=False, random_state=42)

        df_sim = pd.concat([df_sim_yes, df_sim_no], axis=0, ignore_index=True)
        
        df_sim = df_sim[['src', 'src_id', 'dest', 'dest_id', 'is_test', 'score', 'similar']]
        df_sim_reverse = df_sim[['dest', 'dest_id', 'src', 'src_id', 'is_test', 'score', 'similar']].copy()
        df_sim_reverse.columns = ['src', 'src_id', 'dest', 'dest_id', 'is_test', 'score', 'similar']
        df_sim = pd.concat([df_sim, df_sim_reverse], axis=0, ignore_index=True)
        self.df_sim = df_sim.drop_duplicates(subset=['src', 'dest'], keep='last', ignore_index=True)
        
        # close connections
        cur.close()
        con.close()
        
    def __len__(self):
        return self.df_sim.shape[0]

    def __getitem__(self, index):
        # ((track 1, track 2), similar (1/0))
        src_id = int(self.df_sim.loc[index, 'src_id'])
        dest_id = int(self.df_sim.loc[index, 'dest_id'])
        return ((self.data[src_id], self.data[dest_id]), self.df_sim.loc[index, 'similar'])