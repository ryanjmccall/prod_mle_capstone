from typing import Tuple

import numpy as np
import pandas as pd
import prefect
from prefect import task
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.callbacks import TimerCallback

from sentiment_classifier.pipeline import get_train_pipeline


@task(name='train_model', log_stdout=True)
def train_test_model(df: pd.DataFrame, dag_conf: dict) -> Tuple[BaseEstimator, dict]:
    """Create test and train set, train model, and evaluate on test set.
    
    :param df: MELD dataset
    :param dag_conf: ETL/DAG configuration
    :return: Training pipeline, scoring results
    :rtype: (Pipeline, dict)
    """
    logger = prefect.context.get('logger')

    X = np.array([x for x in df['features']])
    y = df.negativity
    logger.info('X shape %s y shape %s', X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=dag_conf['test_size'],
        stratify=y,
        random_state=dag_conf['random_state']
    )
    logger.info('Train items: %s, Test items: %s', len(y_train), len(y_test))

    pipe = get_train_pipeline(dag_conf)
    pipe.fit(X_train, y_train)
    train_score = f1_score(y_train, pipe.predict(X_train), average='weighted')
    test_score = f1_score(y_test, pipe.predict(X_test), average='weighted')
    logger.info('Train f1 score %s, Test f1 score %s', train_score, test_score)

    metrics = {'f1_score': {'train': train_score, 'test': test_score}}
    return pipe, metrics


@task(name='run_bayes_search', log_stdout=True)
def run_bayes_search(df: pd.DataFrame, dag_conf: dict) -> Tuple[BaseEstimator, dict]:
    """Run hyperparameter search guided by bayes optimization method.

    :param df: MELD dataset
    :param dag_conf: ETL/DAG configuration
    :return: Training pipeline, best model params
    :rtype: (Pipeline, dict)
    """
    logger = prefect.context.get('logger')

    X = np.array([x for x in df['features']])
    y = df.negativity
    cbs = [TimerCallback()]
    pipe = get_train_pipeline(dag_conf)
    search = BayesSearchCV(estimator=pipe, **dag_conf['bayes_search'])

    search.fit(X, y, callback=cbs)

    iter_times = cbs[0].iter_time
    logger.info(f'Search elapsed time: {sum(iter_times) / 60:.2f}m ave-iter={np.mean(iter_times):.2f}s')
    # TODO fix UT mocking and remove hasattr
    params = search.best_params_ if hasattr(search, 'best_params_') else dict()
    # TODO does BayesSearchCV contain the best model after search.fit? doubtful, probably want to set it
    # before returning the pipeline
    return pipe, params
