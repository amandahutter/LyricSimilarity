import unittest
from lyric_sim.dataloaders import MusixMatchDataset

dataset = MusixMatchDataset('./files/mxm_dataset_test.txt')

class TestMxmDataset(unittest.TestCase):

    def test_num_classes(self):
        self.assertEqual(len(dataset.get_classes()), 5000)

    def test_get_item(self):
        self.assertEqual(dataset.__get_item__(0).shape[0], 5000)


if __name__ == '__main__':
    unittest.main()