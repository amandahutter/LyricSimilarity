import unittest
from lyric_sim.datasets import MusixMatchDataset

dataset = MusixMatchDataset('./test_files/mxm_dataset_test.txt')

class TestMxmDataset(unittest.TestCase):

    def test_num_words(self):
        self.assertEqual(len(dataset.get_words()), 5000)

    def test_getitem(self):
        input, label = dataset.__getitem__(0)
        self.assertEqual(input.shape[0], 5000)
        self.assertEqual(label, 'TRAABRX12903CC4816')
    
    def test_getlen(self):
        self.assertEqual(dataset.__len__(), 4)
    
    def test_lyric_index(self):
        input, _ = dataset.__getitem__(0)
        self.assertEqual(input[1], 19)


if __name__ == '__main__':
    unittest.main()