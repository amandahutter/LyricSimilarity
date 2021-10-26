import unittest
from lyric_sim.datasets import MusixMatchDataset

dataset = MusixMatchDataset('./test_files/mxm_dataset_test.txt')

class TestMxmDataset(unittest.TestCase):

    def test_num_words(self):
        self.assertEqual(len(dataset.get_words()), 5000)

    def test_getitem(self):
        self.assertEqual(dataset.__getitem__(0).shape[0], 5000)


if __name__ == '__main__':
    unittest.main()