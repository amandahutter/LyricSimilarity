import unittest
from lyric_sim.datasets.mxm import MusixMatchCsvDataset, MusixMatchSqliteDataset

csv_dataset = MusixMatchCsvDataset('./test_files/mxm_dataset_test.txt')
sqlite_dataset = MusixMatchSqliteDataset('./test_files/mxm_dataset_test.db', './test_files/lastfm_similars_test.db', False, False)

class TestMxmSqliteDataset(unittest.TestCase):

    def test_num_words(self):
        self.assertEqual(len(sqlite_dataset.get_words()), 5000)

    def test_getitem(self):
        input, label = sqlite_dataset.__getitem__(0)
        self.assertEqual(input.shape[0], 5000)
        self.assertEqual(label, 'TRAAAAV128F421A322')
    
    def test_getlen(self):
        self.assertEqual(sqlite_dataset.__len__(), 23)
    
    def test_lyric_index(self):
        input, _ = sqlite_dataset.__getitem__(0)
        self.assertEqual(input[0], 6)

class TestMxmCsvDataset(unittest.TestCase):

    def test_num_words(self):
        self.assertEqual(len(csv_dataset.get_words()), 5000)

    def test_getitem(self):
        input, label = csv_dataset.__getitem__(0)
        self.assertEqual(input.shape[0], 5000)
        self.assertEqual(label, 'TRAABRX12903CC4816')
    
    def test_getlen(self):
        self.assertEqual(csv_dataset.__len__(), 128)
    
    def test_lyric_index(self):
        input, _ = csv_dataset.__getitem__(0)
        self.assertEqual(input[1], 19)


if __name__ == '__main__':
    unittest.main()