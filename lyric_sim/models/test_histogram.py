import unittest
from lyric_sim.datasets import MusixMatchDataset
from lyric_sim.models.histogram import HistogramModel

dataset = MusixMatchDataset('./test_files/mxm_dataset_test.txt')

class TestMxmHistogram(unittest.TestCase):

    def test_forward(self):
        x = dataset.__getitem__(0)
        model = HistogramModel(len(dataset.get_words()), 4096)
        y = model.forward(x)
        self.assertEqual(len(y), 1)


if __name__ == '__main__':
    unittest.main()