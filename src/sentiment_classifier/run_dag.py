import argparse
from importlib import import_module
from typing import Tuple

import numpy as np
import pandas as pd
import prefect
from prefect import case, task, Flow, Parameter
from prefect.executors import LocalDaskExecutor
from prefect.tasks.control_flow import merge
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from skopt import BayesSearchCV
from skopt.callbacks import TimerCallback

from sentiment_classifier.pipeline import get_train_pipeline
from sentiment_classifier.task.extract import extract_features_task
from sentiment_classifier.context import DATA_DIR, DF_DIR
from sentiment_classifier.task.meld_wrangling import convert_mp4_to_wav, load_labels, add_audio_to_labels
from sentiment_classifier.task.checkpoint import checkpoint_exists, load_checkpoint, write_checkpoint
from sentiment_classifier.task.record import record_results


@task(name='prepare_data', log_stdout=True)
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    logger = prefect.context.get('logger')

    before = len(df)
    df.dropna(inplace=True)
    logger.info('Dropped na changed items from %s -> %s', before, len(df))

    # create binary negativity variable
    df['negativity'] = df.Sentiment.apply(lambda x: 1 if x.lower() == 'negative' else 0)
    logger.info('Added negativity to df')
    return df


@task(name='run_bayes_search', log_stdout=True)
def run_bayes_search(df: pd.DataFrame, conf: dict) -> Tuple[BaseEstimator, dict]:
    logger = prefect.context.get('logger')

    X = np.array([x for x in df['features']])
    y = df.negativity
    cbs = [TimerCallback()]
    pipe = get_train_pipeline(conf)
    search = BayesSearchCV(estimator=pipe, **conf['bayes_search'])

    search.fit(X, y, callback=cbs)

    iter_times = cbs[0].iter_time
    logger.info(f'Search elapsed time: {sum(iter_times) / 60:.2f}m ave-iter={np.mean(iter_times):.2f}s')
    # TODO fix UT mocking and remove hasattr
    params = search.best_params_ if hasattr(search, 'best_params_') else dict()
    # TODO does BayesSearchCV contain the best model after search.fit? doubtful, probably want to set it
    # before returning the pipeline
    return pipe, params


@task(name='train_model', log_stdout=True)
def train_test_model(df: pd.DataFrame, conf: dict) -> Tuple[BaseEstimator, dict]:
    logger = prefect.context.get('logger')

    X = np.array([x for x in df['features']])
    y = df.negativity
    logger.info('X shape %s y shape %s', X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=conf['test_size'],
        stratify=y,
        random_state=conf['random_state']
    )
    logger.info('Train items: %s, Test items: %s', len(y_train), len(y_test))

    pipe = get_train_pipeline(conf)
    pipe.fit(X_train, y_train)
    train_score = f1_score(y_train, pipe.predict(X_train), average='weighted')
    test_score = f1_score(y_test, pipe.predict(X_test), average='weighted')
    logger.info('Train f1 score %s, Test f1 score %s', train_score, test_score)

    metrics = {'f1_score': {'train': train_score, 'test': test_score}}
    return pipe, metrics


def get_flow(dag_config: dict, run_search: bool) -> Flow:
    with Flow('sentiment_classifier_ETL') as flow:
        # Specify ETL DAG to prefect within context manager

        # Define constants
        data_dir = Parameter('data_dir', default=DATA_DIR)
        df_dir = Parameter('df_dir', default=DF_DIR)
        dag_config = Parameter('dag_config', default=dag_config)
        run_search = Parameter('run_search', default=run_search)

        # Build DAG
        df_checkpoint = checkpoint_exists(df_dir)
        with case(df_checkpoint, False):
            # Rebuild the dataset Dataframe from original MELD data
            wavs_dirs = convert_mp4_to_wav(data_dir)
            labels_dfs = load_labels(data_dir)
            df_true = add_audio_to_labels(wavs_dirs, labels_dfs)
            df_true = prepare_data(df_true)
            df_true = extract_features_task(df_true, dag_config)
            write_checkpoint(df_true, df_dir)

        with case(df_checkpoint, True):
            # Load dataset Dataframe from checkpoint
            df_false = load_checkpoint(df_dir)

        # Join rebuild-load workflows
        df = merge(df_true, df_false)

        # Now either run hyperparam search or train/test a model
        with case(run_search, True):
            train_pipe_1, metadata_1 = run_bayes_search(df, dag_config)

        with case(run_search, False):
            train_pipe_2, metadata_2 = train_test_model(df, dag_config)

        train_pipe = merge(train_pipe_1, train_pipe_2)
        metadata = merge(metadata_1, metadata_2)
        record_results(data_dir, train_pipe, dag_config, metadata)

    return flow


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        help='Python path to DAG config. config dict expected to be named DAG_CONFIG.',
        default='sentiment_classifier.config.default_config'
    )
    parser.add_argument(
        '-s', '--search',
        help='run model hyperparameter search instead of model training. '
             'will not output production-ready prediction pipeline',
        action='store_true'
    )
    return parser.parse_args()


def main():
    args = get_args()
    config_module = import_module(args.config)
    dag_config = getattr(config_module, 'DAG_CONFIG')

    # Use threads since the prefect tasks make use of libraries like numpy, pandas,
    # and scikit-learn that release the GIL
    executor = LocalDaskExecutor(scheduler='threads')
    flow = get_flow(dag_config=dag_config, run_search=args.search)
    flow.run(executor=executor)


if __name__ == '__main__':
    main()
