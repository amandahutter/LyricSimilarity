import unittest
from lyric_sim.datasets.mxm import MxMLastfmJoinedDataset
from lyric_sim.models.histogram import HistogramModel
import torch

dataset = MxMLastfmJoinedDataset('./test_files/mxm_lastfm.db', False)
class TestMxmHistogram(unittest.TestCase):

    def test_forward(self):
        example, _ = dataset.__getitem__(0)
        
        # convert inputs into batches of size N=1
        example = torch.unsqueeze(example, 0)

        model = HistogramModel(10000, 8192)
        y = model.forward(example)
        self.assertEqual(len(y), 1)


if __name__ == '__main__':
    unittest.main()