import unittest

import pandas as pd

from sentiment_classifier.task.extract import extract_features


class TestExtract(unittest.TestCase):

    def test_extract_features_task(self):
        df = pd.DataFrame([])
        conf = dict()

        # TODO
        df = extract_features.run(df, conf)
        # assert df.features

    def test_extract_features_task_bad_data(self):
        pass
        # TODO
