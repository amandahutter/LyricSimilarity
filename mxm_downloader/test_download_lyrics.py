import unittest
from mxm_downloader.download_lyrics import strip_warning, lyrics_response_to_tuple
from mxm_downloader.test_contants import EXAMPLE_LYRICS, EXAMPLE_LYRICS_STRIPPED, LYRICS_RESPONSE, LYRICS_TUPLE, MXM_ID


class TestMxmDownloader(unittest.TestCase):
    def test_strip_warning(self):
        self.assertEqual(strip_warning(EXAMPLE_LYRICS), EXAMPLE_LYRICS_STRIPPED)

    def test_lyrics_response_to_tuple(self):
        self.assertEqual(lyrics_response_to_tuple(MXM_ID, LYRICS_RESPONSE), LYRICS_TUPLE)

if __name__ == '__main__':
    unittest.main()