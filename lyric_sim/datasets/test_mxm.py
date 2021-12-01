import unittest
from lyric_sim.datasets.mxm import MusixMatchCsvDataset, MxMLastfmJoinedDataset

class TestMxMLastfmJoinedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestMxMLastfmJoinedDataset, self).setUpClass()
        self.dataset = MxMLastfmJoinedDataset('./test_files/mxm_lastfm.db', False)

    def test_getitem(self):
        input, label = self.dataset.__getitem__(0)
        self.assertEqual(input.shape[0], 10000)
        self.assertEqual(label, 0)
    
    def test_getlen(self):
        self.assertEqual(self.dataset.__len__(), 29)

class TestMxmCsvDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        super(TestMxmCsvDataset, self).setUpClass()
        self.csv_dataset = MusixMatchCsvDataset('./test_files/mxm_dataset.txt')

    def test_num_words(self):
        self.assertEqual(len(self.csv_dataset.get_words()), 5000)

    def test_getitem(self):
        input, label = self.csv_dataset.__getitem__(0)
        self.assertEqual(input.shape[0], 5000)
        self.assertEqual(label, 'TRAABRX12903CC4816')
    
    def test_getlen(self):
        self.assertEqual(self.csv_dataset.__len__(), 128)
    
    def test_lyric_index(self):
        input, _ = self.csv_dataset.__getitem__(0)
        self.assertEqual(input[1], 19)


if __name__ == '__main__':
    unittest.main()