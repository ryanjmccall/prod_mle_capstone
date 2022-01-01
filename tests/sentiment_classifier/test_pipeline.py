import unittest

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import sentiment_classifier.pipeline as pipe


class TestPipeline(unittest.TestCase):

    def test_prepare_data_task(self):
        df = pd.DataFrame(['Negative', None, 'positive', None],
                          columns=['Sentiment'])

        result = pipe.prepare_data.run(df)

        assert list(result.negativity) == [1, 0]
        assert list(result.Sentiment) == ['Negative', 'positive']

    def test_run_bayes_search_task(self):
        df = pd.DataFrame([])
        conf = dict()
        # TODO

        result = pipe.run_bayes_search.run(df, conf)

    def test_train_test_model_task(self):
        df = pd.DataFrame([])
        conf = dict()
        # TODO

        model, metrics = pipe.train_test_model.run(df, conf)

    def test_get_ml_pipeline(self):
        conf = dict()
        # TODO

        ml_pipeline = pipe.get_ml_pipeline(conf)

    def test_eval_parallel_pred_task(self):
        model = RandomForestClassifier()
        df = pd.DataFrame([])
        # TODO

        elapsed, predictions = pipe.eval_parallel_pred.run(model, df)

    def test_record_results_task(self):
        data_dir = ''
        model = RandomForestClassifier()
        config = dict()
        metrics = dict()

        pipe.record_results.run(data_dir, model, config, metrics)

        # TODO assert files are stored in data_dir

    # TODO error cases
