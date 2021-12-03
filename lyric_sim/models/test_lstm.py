import unittest 
from lyric_sim.models.lstm import LSTM, CombinationType
import torch 
import numpy as np 


class TestLyricsLSTM(unittest.TestCase):

    def test_forward(self):
        

        vocab_size = 1000
        song1 = torch.randint(low = 0, high = vocab_size, size = (1,20))
        song2 = torch.randint(low = 0, high = vocab_size, size = (1,20))
        similar = 1 

        model = LSTM(vocab_size, 20, 10, 0.9, 1, CombinationType.ADD) 
        y = model.forward(song1, song2)
        self.assertEqual(len(y), 1)

        model = LSTM(vocab_size, 20, 10, 0.9, 1, CombinationType.SUB) 
        y = model.forward(song1, song2)
        self.assertEqual(len(y), 1)

        model = LSTM(vocab_size, 20, 10, 0.9, 1, CombinationType.MULT) 
        y = model.forward(song1, song2)
        self.assertEqual(len(y), 1)

        model = LSTM(vocab_size, 20, 10, 0.9, 1, CombinationType.ALL) 
        y = model.forward(song1, song2)
        self.assertEqual(len(y), 1)

if __name__ == '__main__':
    unittest.main()
