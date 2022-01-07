import os
import shutil
import unittest

from sentiment_classifier.pipeline import get_train_pipeline
from sentiment_classifier.task.record import record_results
from sentiment_classifier.config.default_config import DAG_CONFIG
from sentiment_classifier.context import ROOT_DIR


class TestRecord(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_path = os.path.join(ROOT_DIR, '..', '..', 'tests', 'test_data')

    def test_record_results_task(self):
        pipe = get_train_pipeline(DAG_CONFIG)
        config = DAG_CONFIG
        metadata = dict()

        out_path = record_results.run(self.test_data_path, pipe, config, metadata)

        assert list(sorted(os.listdir(out_path))) == ['prediction_pipeline.joblib', 'record.json']
        shutil.rmtree(out_path)
