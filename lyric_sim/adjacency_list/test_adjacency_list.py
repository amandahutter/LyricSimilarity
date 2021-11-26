import unittest
from lyric_sim.adjacency_list.adjacency_list import AdjacencyList, SongNotFoundException

adjacency_list = AdjacencyList('./test_files/lastfm_similars_test.db', './test_files/mxm_dataset_test.db', False)

class TestAdjacencyList(unittest.TestCase):

    def test_get_adjacency(self):
        src_id = 'TRCCCYE12903CFF0E9'
        dest_id = 'TRHZRQH128F92F9AC2'
        similarity = adjacency_list.get_similarity(src_id, dest_id)
        self.assertEqual(similarity, 0.498053)
    
    def test_song_not_found(self):
        try:
            adjacency_list.get_similarity('not_an_id', 'unused')
        except SongNotFoundException as ex:
           self.assertEqual(str(ex), 'Song with mxm id not_an_id not found in adjacency list')
    
    def test_not_adjecent_returns_zero(self):
        src_id = 'TRCCCYE12903CFF0E9'
        similarity = adjacency_list.get_similarity(src_id, 'not_an_id')
        self.assertEqual(similarity, 0)

if __name__ == '__main__':
    unittest.main()