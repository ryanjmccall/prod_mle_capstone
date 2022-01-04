import os
import shutil
import unittest

import numpy as np
import pandas as pd

from sentiment_classifier.context import DATA_DIR, ROOT_DIR
from sentiment_classifier.task.meld_wrangling import convert_mp4_to_wav, load_labels, add_audio_to_labels


class TestMeldWrangling(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_path = os.path.join(ROOT_DIR, '..', '..', 'tests', 'test_data')
        cls.test_audio_path = os.path.join(cls.test_data_path, 'audio')
        cls.test_fname = '10_5.wav'

    def tearDown(self) -> None:
        if os.path.exists(self.test_audio_path):
            shutil.rmtree(self.test_audio_path)

    def test_convert_mp4_to_wav_task(self):
        dest_paths = convert_mp4_to_wav.run(self.test_data_path)

        for path in dest_paths:
            assert os.listdir(path) == [self.test_fname], path
            assert os.path.getsize(os.path.join(path, self.test_fname)) > 0, path

    def test_load_labels_task(self):
        train, dev, test = load_labels.run(DATA_DIR)

        assert train.shape == (9989, 12)
        assert dev.shape == (1109, 12)
        assert test.shape == (2610, 12)
        assert 'dia_utt' in train
        assert train.dia_utt.head(1).values == '0_0'
        assert 'dia_utt' in dev
        assert dev.dia_utt.head(1).values == '0_0'
        assert 'dia_utt' in test
        assert test.dia_utt.head(1).values == '0_0'

    def test_add_audio_to_labels_task(self):
        wav_dirs = convert_mp4_to_wav.run(self.test_data_path)
        # 1 uid having a wav and 1 uid missing a wav
        label_dfs = [pd.DataFrame(['10_5', '0_0'], columns=['dia_utt']) for _ in range(3)]

        df = add_audio_to_labels.run(wav_dirs, label_dfs)

        assert len(df) == 6
        assert list(df.dia_utt.values) == ['10_5', '0_0'] * 3
        srs = list(df.sr.values)
        for i in (0, 2, 4):
            assert srs[i] == 22050.0, i
        for i in (1, 3, 5):
            assert np.isnan(srs[i])
