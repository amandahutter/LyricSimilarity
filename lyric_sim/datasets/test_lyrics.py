import unittest
from lyric_sim.datasets.lyrics import LyricsSqliteDataset

lyrics_db_path = '../test_files/mxm_lyrics.db'
mxm_db_path = '../test_files/mxm_dataset.db'
lastfm_db_path = '../test_files/mxm_lastfm.db'

dataset = LyricsSqliteDataset(lyrics_db_path, mxm_db_path, lastfm_db_path, False, True)

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