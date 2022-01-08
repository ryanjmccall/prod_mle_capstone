import os
import shutil
import unittest

from sklearn.datasets import make_classification

from sentiment_classifier.pipeline import get_train_pipeline
from sentiment_classifier.task.record import record_results
from sentiment_classifier.config.default_config import DAG_CONFIG
from sentiment_classifier.context import ROOT_DIR


class TestRecord(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_path = os.path.join(ROOT_DIR, '..', '..', 'tests', 'test_data')

    def test_record_results_task(self):
        config = DAG_CONFIG
        pipe = get_train_pipeline(config)
        model = pipe[-1]
        X, y = make_classification(n_samples=10, n_features=10, n_classes=2)
        model.fit(X, y)
        metadata = dict()

        out_path = record_results.run(self.test_data_path, pipe, config, metadata)

        assert list(sorted(os.listdir(out_path))) == ['prediction_pipeline.joblib', 'record.json']
        shutil.rmtree(out_path)
