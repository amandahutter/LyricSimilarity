MAX_LENGTH = 150
SIM_THRESHOLD = 0.5

class LyricsSqliteDataset(Dataset):
    def __init__(self, lyrics_db_path: str, mxm_db_path: str, lastfm_db_path: str, use_test: bool, pad_length: bool):
        use_test = int(use_test)
        
        # establish connection to lyrics
        con = sqlite3.connect(lyrics_db_path)
        
        # establish the other connections we need
        con_mxm = sqlite3.connect(mxm_db_path)
        con_lastfm = sqlite3.connect(lastfm_db_path)

        # read in lyrics
        qry = "SELECT * FROM lyrics WHERE lyrics <> ''"
        df = pd.read_sql_query(qry, con)

        # read in mxm (ids only)
        qry = 'SELECT DISTINCT track_id AS tid, mxm_tid AS mxm_id FROM lyrics WHERE is_test = {}'.format(use_test)
        df_mxm_ids = pd.read_sql_query(qry, con_mxm)

        # similarity score (from lastfm)
        qry = 'SELECT src, dest, score, is_test FROM similars_src WHERE is_test = {}'.format(use_test)
        df_sim_all = pd.read_sql_query(qry, con_lastfm)

        # close connections
        con.close()
        con_mxm.close()
        con_lastfm.close()

        # join ids in to lyrics data 
        df = df.merge(df_mxm_ids, how='inner', on='mxm_id')

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
        vocab = build_vocab_from_iterator(map(tokenizer, df['lyrics']), specials=['<pad>'])
        vocab.set_default_index(0)
        self.vocab = vocab
        self.data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in df['lyrics']]
        
        # merge in ids to similarity data
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

        self.df_sim = pd.concat([df_sim_yes, df_sim_no], axis=0, ignore_index=True)

        
    def __len__(self):
        return self.df_sim.shape[0]

    def __getitem__(self, index):
        # ((track 1, track 2), similar (1/0))
        src_id = int(self.df_sim.loc[index, 'src_id'])
        dest_id = int(self.df_sim.loc[index, 'dest_id'])
        return ((self.data[src_id], self.data[dest_id]), self.df_sim.loc[index, 'similar'])