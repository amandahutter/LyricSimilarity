import unittest
from lyric_sim.datasets.mxm import LyricsSqliteDataset

lyrics_db_path = '../data_files/mxm_lyrics.db'
mxm_db_path = '../data_files/mxm_dataset.db'
lastfm_db_path = '../data_files/lastfm_similars.db'
#sim_csv_path = '../data_files/Similarity_df.csv'
sim_csv_path = '../data_files/Similarity_df_mini.csv' #sampled similarity df

dataset = LyricsSqliteDataset(lyrics_db_path, mxm_db_path, lastfm_db_path, sim_csv_path, False, False)

class TestLyricsSqliteDataset(unittest.TestCase):

    def test_vocab(self):
        word_id = dataset.vocab.__getitem__('wheel')
        self.assertEqual(word_id, 1480)

    def test_getitem(self):
        lyrics, label = dataset.__getitem__(0)
        self.assertEqual(len(lyrics), 2)
        self.assertEqual(label, 1)
    
    def test_getlen(self):
        self.assertEqual(dataset.__len__(), 6466)

if __name__ == '__main__':
    unittest.main()