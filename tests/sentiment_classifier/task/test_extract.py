import unittest
from unittest.mock import patch, call

import numpy as np
import pandas as pd

from sentiment_classifier.task.extract import extract_features_task, extract_melspectrogram, extract_mfcc, extract_chroma


class TestExtract(unittest.TestCase):

    def setUp(self) -> None:
        self.data = [[list(range(100)), 10],
                     [list(reversed(range(100))), 10]]
        self.X = np.array(range(1000), dtype=np.float32)
        self.conf = dict(extract=dict(audio_limit=2,
                                      mel_window_length=1,
                                      n_mels=2,
                                      mfcc_window_length=2,
                                      n_mfcc=3,
                                      chroma_window_length=4,
                                      n_chroma=5))

    @patch('sentiment_classifier.task.extract.extract_chroma')
    @patch('sentiment_classifier.task.extract.extract_mfcc')
    @patch('sentiment_classifier.task.extract.extract_melspectrogram')
    def test_extract_features_task(self, mock_mel, mock_mfcc, mock_chroma):
        mock_mel.return_value = [1]
        mock_mfcc.return_value = [2]
        mock_chroma.return_value = [3]
        df = pd.DataFrame(self.data, columns=['audio', 'sr'])

        df = extract_features_task.run(df, self.conf)

        mock_mel.assert_has_calls(
            [call([0, 1], 10, 1, 2),
             call([99, 98], 10, 1, 2)]
        )
        mock_mfcc.assert_has_calls(
            [call([0, 1], 10, 2, 3),
             call([99, 98], 10, 2, 3)]
        )
        mock_chroma.assert_has_calls(
            [call([0, 1], 10, 4, 5),
             call([99, 98], 10, 4, 5)]
        )
        assert 'features' in df
        assert len(df.features.values) == 2
        assert df.features.values[0].tolist() == [1, 2, 3]
        assert df.features.values[1].tolist() == [1, 2, 3]

    def test_extract_features_task_no_audio_raises_error(self):
        df = pd.DataFrame([[None, None]], columns=['audio', 'sr'])

        with self.assertRaises(ValueError):
            extract_features_task.run(df, self.conf)

    def test_extract_melspectrogram(self):
        result = extract_melspectrogram(self.X, sr=10, win_length=32, n_mels=128)

        assert result.shape == (128,)

    def test_extract_mfcc(self):
        result = extract_mfcc(self.X, sr=10, win_length=32, n_mfcc=20)

        assert result.shape == (20,)

    def test_extract_chroma(self):
        result = extract_chroma(self.X, sr=10, win_length=32, n_chroma=12)

        assert result.shape == (12,)
