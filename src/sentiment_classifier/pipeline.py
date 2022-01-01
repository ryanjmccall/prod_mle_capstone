import argparse
from datetime import datetime
import json
import os
import time
from typing import Tuple

import dask.array as da
import numpy as np
from dask_ml.wrappers import ParallelPostFit
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from joblib import dump
import lightgbm as lgb
import pandas as pd
import prefect
from prefect import case, task, Flow, Parameter
from prefect.executors import LocalDaskExecutor
from prefect.tasks.control_flow import merge
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from skopt import BayesSearchCV
from skopt.callbacks import TimerCallback

from sentiment_classifier.task.extract import extract_features
from sentiment_classifier.config import DAG_CONFIG
from sentiment_classifier.context import DATA_DIR, DF_DIR
from sentiment_classifier.task.meld_wrangling import convert_mp4_to_wav, load_labels, add_audio_to_labels
from sentiment_classifier.task.checkpoint import checkpoint_exists, load_checkpoint, write_checkpoint


@task(name='prepare_data', log_stdout=True)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    log = prefect.context.get('logger')
    before = len(df)
    df.dropna(inplace=True)
    log.info('dropped na changed items from %s -> %s', before, len(df))

    # create binary negativity variable
    df['negativity'] = df.Sentiment.apply(lambda x: 1 if x.lower() == 'negative' else 0)
    return df


@task(name='run_bayes_search', log_stdout=True)
def run_bayes_search(df: pd.DataFrame, conf: dict) -> None:
    log = prefect.context.get('logger')

    X = df.features
    y = df.negativity
    cbs = [TimerCallback()]
    pipe = get_ml_pipeline(conf)
    opt = BayesSearchCV(estimator=pipe, **conf['bayes_search'])

    search = opt.fit(X, y, callback=cbs)

    iter_times = cbs[0].iter_time
    log.info(f'elapsed time: {sum(iter_times) / 60:.2f}m ave-iter={np.mean(iter_times):.2f}s')
    log.info('best search params: %s', search.best_params_)


@task(name='train_model', log_stdout=True)
def train_test_model(df: pd.DataFrame, conf: dict) -> Tuple[BaseEstimator, dict]:
    log = prefect.context.get('logger')
    X = df.features
    y = df.negativity
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=conf['test_size'],
        stratify=y,
        random_state=conf['random_state']
    )
    log.info('Train items: %s, Test items: %s', len(y_train), len(y_test))

    pipe = get_ml_pipeline(conf)
    pipe.fit(X_train, y_train)
    train_score = f1_score(y_train, pipe.predict(X_train), average='weighted')
    test_score = f1_score(y_test, pipe.predict(X_test), average='weighted')

    log.info('Train f1 score %s, Test f1 score %s', train_score, test_score)

    metrics = {'f1_score': {'train': train_score, 'test': test_score}}
    return pipe['model'], metrics


def get_ml_pipeline(conf: dict) -> Pipeline:
    return Pipeline([
        ('standardize', QuantileTransformer(**conf['standardize'])),
        ('decomposition', PCA(**conf['decomposition'])),
        ('oversample', ADASYN(**conf['oversample'])),
        ('model', lgb.LGBMClassifier(**conf['model'])),
    ])


@task(name='evaluate_parallel_prediction', log_stdout=True)
def eval_parallel_pred(model, df, duplication=5) -> Tuple[float, int]:
    """Evaluate parallelized prediction using Dask."""
    log = prefect.context.get('logger')
    X = df.features
    y = df.negativity
    parallel_model = ParallelPostFit(model, scoring='accuracy')
    parallel_model.fit(X, y)

    X_large = dupe_data(X, duplication)
    y_large = dupe_data(y, duplication)

    start = time.time()
    model.score(X_large, y_large)
    finish = time.time()
    elapsed = finish - start
    predictions = X_large.shape[0]

    log.info(f'elapsed  {elapsed:.3f} s')
    log.info(f'items    {predictions:,}')
    log.info(f'per item {elapsed / predictions * 1e6:.3f} us')
    return elapsed, predictions


def dupe_data(X, duplication: int):
    return da.concatenate([da.from_array(X.to_numpy(), chunks=X.shape)
                           for _ in range(duplication)])


@task(name='record_results', log_stdout=True)
def record_results(data_dir: str, model: BaseEstimator, dag_config: dict, metrics: dict) -> None:
    """Save model, config, and metrics of the current DAG run."""
    log = prefect.context.get('logger')
    dt = str(datetime.now())
    results_path = os.path.join(data_dir, 'results', dt)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    log.info('Recording results to %s', results_path)
    dump(model, filename=os.path.join(results_path, 'model.joblib'))
    record = dict(dag_config=dag_config, metrics=metrics)
    with open(os.path.join(results_path, 'record.json'), 'w', encoding='utf8') as f:
        json.dump(record, f)


def get_flow(args) -> Flow:
    with Flow('sentiment_classifier_ETL') as flow:
        # Specify ETL DAG to prefect within context manager

        # Define constants
        data_dir = Parameter('data_dir', default=DATA_DIR)
        df_dir = Parameter('df_dir', default=DF_DIR)
        dag_config = Parameter('dag_config', default=DAG_CONFIG)  # could be specified by clarg
        run_search = Parameter('run_search', default=args.search)

        # Build DAG
        df_checkpoint = checkpoint_exists(df_dir)
        with case(df_checkpoint, False):
            # Rebuild the dataset Dataframe from original MELD data
            wavs_dirs = convert_mp4_to_wav(data_dir)
            labels_dfs = load_labels(data_dir)
            df_true = add_audio_to_labels(wavs_dirs, labels_dfs)
            df_true = extract_features(df_true, dag_config)
            write_checkpoint(df_true, df_dir)

        with case(df_checkpoint, True):
            # Load dataset Dataframe from checkpoint
            df_false = load_checkpoint(df_dir)

        # Join rebuild-load workflows
        df = merge(df_true, df_false)

        # Now either run hyperparam search or train/test a model
        df = prepare_data(df)
        with case(run_search, True):
            run_bayes_search(df, dag_config)

        with case(run_search, False):
            model, metrics = train_test_model(df, dag_config)
            eval_parallel_pred(model, df)
            record_results(data_dir, model, dag_config, metrics)

    return flow


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--search',
        help='run model hyperparameter search instead of model training',
        action='store_true'
    )
    return parser.parse_args()


def run_pipeline():
    args = get_args()

    # Use threads since these tasks make use of libraries like numpy, pandas,
    # amd scikit-learn that release the GIL
    executor = LocalDaskExecutor(scheduler='threads')
    flow = get_flow(args)
    flow.run(executor=executor)


if __name__ == '__main__':
    run_pipeline()
