import unittest

from sentiment_classifier.task.meld_wrangling import convert_mp4_to_wav, load_labels, add_audio_to_labels


class TestMeldWrangling(unittest.TestCase):
    def test_convert_mp4_to_wav_task(self):
        data_dir = ''
        # TODO generate a fake directory to run this test with 1 file in train, dev, test

        dest_paths = convert_mp4_to_wav.run(data_dir)

    def test_load_labels_task(self):
        data_dir = ''

        train, dev, test = load_labels.run(data_dir)

        # TODO

    def test_add_audio_to_labels_task(self):
        wav_dirs = []
        label_dfs = []

        df = add_audio_to_labels.run(wav_dirs, label_dfs)

        # TODO

    # TODO error cases