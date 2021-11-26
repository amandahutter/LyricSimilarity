import unittest
from lyric_sim.datasets import MusixMatchDataset
from lyric_sim.models.histogram import HistogramModel
import torch

dataset = MusixMatchDataset('./test_files/mxm_dataset_test.txt')
class TestMxmHistogram(unittest.TestCase):

    def test_forward(self):
        src, label = dataset.__getitem__(0)
        dest, label = dataset.__getitem__(1)
        
        # convert inputs into batches of size N=1
        src = torch.unsqueeze(src, 0)
        dest = torch.unsqueeze(dest, 0)
        model = HistogramModel(len(dataset.get_words()), 8192)
        y = model.forward(src, dest)
        self.assertEqual(len(y), 1)


if __name__ == '__main__':
    unittest.main()