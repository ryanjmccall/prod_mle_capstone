import os
import unittest
from copy import deepcopy
from unittest.mock import patch

import pandas as pd
import skopt
from imblearn.over_sampling import ADASYN
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from skopt.space import Integer

from sentiment_classifier.run_dag import prepare_data, run_bayes_search, \
    train_test_model
from sentiment_classifier.pipeline import get_train_pipeline, get_prediction_pipeline
from sentiment_classifier.config.default_config import DAG_CONFIG
from sentiment_classifier.context import ROOT_DIR


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_data_path = os.path.join(ROOT_DIR, '..', '..', 'tests', 'test_data')
        cls.sample_df = pd.read_pickle(os.path.join(cls.test_data_path, 'features', 'sample.pkl'))

    def test_prepare_data_task(self):
        df = pd.DataFrame(['Negative', None, 'positive', None],
                          columns=['Sentiment'])
        
        result = prepare_data.run(df)

        assert list(result.negativity) == [1, 0]
        assert list(result.Sentiment) == ['Negative', 'positive']

    @patch.object(skopt.BayesSearchCV, 'fit')
    def test_run_bayes_search_task(self, mock_fit):
        conf = deepcopy(DAG_CONFIG)
        conf['bayes_search'] = dict(
            n_iter=1,
            n_jobs=-1,
            refit=True,
            cv=2,
            search_spaces=dict(
                model__num_leaves=Integer(10, 11)
            )
        )

        run_bayes_search.run(self.sample_df, conf)

        mock_fit.assert_called_once()

    def test_train_test_model_task(self):
        model, metrics = train_test_model.run(self.sample_df, DAG_CONFIG)

        assert isinstance(model, BaseEstimator)
        assert 'f1_score' in metrics
        assert 'train' in metrics['f1_score']
        assert metrics['f1_score']['train'] <= 1.0
        assert 'test' in metrics['f1_score']
        assert metrics['f1_score']['test'] <= 1.0

    def test_get_train_pipeline(self):
        pipe = get_train_pipeline(DAG_CONFIG)

        objs = [obj for name, obj in pipe.steps]
        assert len(objs) == 4
        assert isinstance(objs[0], QuantileTransformer)
        assert isinstance(objs[1], PCA)
        assert isinstance(objs[2], ADASYN)
        assert isinstance(objs[3], BaseEstimator)

    def test_get_first_two_steps(self):
        pipe = get_prediction_pipeline(get_train_pipeline(DAG_CONFIG))

        objs = [obj for name, obj in pipe.steps]
        assert len(pipe) == 3
        assert isinstance(objs[0], QuantileTransformer)
        assert isinstance(objs[1], PCA)
        assert isinstance(objs[2], BaseEstimator)
