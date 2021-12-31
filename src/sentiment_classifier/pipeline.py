import argparse
from datetime import datetime
import json
import os
import time
from typing import List, Generator, Tuple

import dask.array as da
from dask_ml.wrappers import ParallelPostFit
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from joblib import dump, load
import librosa
import moviepy.editor as mp
import numpy as np
import pandas as pd
import prefect
from prefect import case, task, Flow, Parameter
from prefect.executors import LocalDaskExecutor
from prefect.tasks.control_flow import merge
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer

from extract import extract_melspectrogram, extract_mfcc, extract_chroma
from sentiment_classifier.config import DAG_CONFIG
from sentiment_classifier.model import get_baked_model


# TODO move to context module
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '../../data')
DF_DIR = os.path.join(DATA_DIR, 'features')
LOG_DIR = os.path.join(ROOT_DIR, '../../logs')
CHECKPOINT_DF_FNAME = 'df_checkpoint.pkl'


# TODO move to util module
def unique_fname(full_path: str) -> str:
    """Standardizes MELD data file name to 'dialog_utterance' form"""
    fname = full_path.split('/')[-1].split('.')[0]
    return fname.replace('dia', '').replace('utt', '').replace('final_videos_test', '')


def listdir_ext(path: str, extension: str) -> Generator[str, None, None]:
    return (f for f in os.listdir(path) if f.split('.')[-1] == extension)


# Convenience function to get int columns (here, features) from a dataframe
# N.B select_dtypes doesn't work b/c that uses cell values, not the dtype of the column name
def get_features(df: pd.DataFrame):
    return df[(c for c in df.columns.values if isinstance(c, int))]


# TODO move tasks to their own module at least
@task(name='extract_wavs', log_stdout=True)
def extract_wavs(data_dir: str) -> List[str]:
    """The MELD dataset provides 3 dirs of .mp4 files. This function converts all .mp4 files
    to .wav and returns the .wav dir. If a .wav file already exists,
    the conversion is skipped.

    Expects the following input directory structure:
    - data_dir/raw/[train|dev|test]
    - data_dir/labels/[train|dev|test]

    Because of file namespace collisions, outputs wavs to following structure:
    - data_dir/audio/[train|dev|test]
    """
    log = prefect.context.get('logger')
    dest_paths = []
    exceptions = []
    converted = 0
    for dir_ in ('train', 'dev', 'test'):
        # N.B. we have duplicate dia_utt ids among the 3 folders, so we have to operate separately
        # until we have all 3 dataframes connecting audio and labels
        dest_path = os.path.join(data_dir, 'audio', dir_)
        dest_paths.append(dest_path)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        finished_fnames = set(f.split('.')[0] for f in listdir_ext(dest_path, 'wav'))
        log.info('Found %s converted .wav files in %s', len(finished_fnames), dest_path)

        src_path = os.path.join(data_dir, 'raw', dir_)
        mp4_paths = (os.path.join(src_path, f) for f in listdir_ext(src_path, 'mp4'))
        for mp4_path in mp4_paths:
            fname = unique_fname(mp4_path)
            if fname in finished_fnames:
                continue

            out_path = os.path.join(data_dir, 'audio', dir_, fname + '.wav')
            try:
                clip = mp.VideoFileClip(mp4_path)

                # highest quality kwargs based on https://zulko.github.io/moviepy/ref/AudioClip.html
                clip.audio.write_audiofile(out_path, nbytes=4, codec='pcm_s32le', bitrate='3000k')
                converted += 1
            except Exception as e:
                # one dataset file was found to be corrupt
                exceptions.append(e)

    log.info('Converted %s files and had %s exceptions(s): \n%s', converted, len(exceptions), exceptions)
    return dest_paths


@task(name='load_data_labels')
def load_data_labels(data_dir: str) -> List[pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(data_dir, 'labels', 'train_labels.csv'))
    dev_df = pd.read_csv(os.path.join(data_dir, 'labels', 'dev_labels.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'labels', 'test_labels.csv'))

    # construct unique ID
    cols = ['Dialogue_ID', 'Utterance_ID']
    train_df['dia_utt'] = train_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    dev_df['dia_utt'] = dev_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    test_df['dia_utt'] = test_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return [train_df, dev_df, test_df]


@task(name='join_audio_to_labels', log_stdout=True)
def join_audio_to_labels(wav_dirs: str, label_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """For each row in labels DataFrame, load corresponding audio and add it to the DataFrame."""
    log = prefect.context.get('logger')
    dfs = []
    for wav_dir, df in zip(wav_dirs, label_dfs):
        log.info('Joining audio from %s to labels df of size %s', wav_dir, len(df))
        audios, srs = [], []
        for uid in df['dia_utt']:
            path = os.path.join(wav_dir, uid + '.wav')
            if os.path.exists(path):
                log.debug('Loading wav %s', path)
                audio, sr = librosa.load(path)
            else:
                log.warning('Wav not found at: %s', path)
                audio = None
                sr = None

            audios.append(audio)
            srs.append(sr)

        df['audio'] = audios
        df['sr'] = srs
        dfs.append(df)

    return pd.concat(dfs)


@task(name='checkpoint_df', log_stdout=True)
def checkpoint_df(df: pd.DataFrame, path: str):
    log = prefect.context.get('logger')
    if not os.path.exists(path):
        os.makedirs(path)

    out_path = os.path.join(path, CHECKPOINT_DF_FNAME)
    log.info('Saving df to %s', out_path)
    df.to_pickle(out_path)


@task(name='base_df_exists', log_stdout=True)
def base_df_exists(path: str) -> bool:
    log = prefect.context.get('logger')
    exists = os.path.exists(os.path.join(path, CHECKPOINT_DF_FNAME))
    log.info('Base df available: %s', exists)
    return exists


@task(name='load_base_df')
def load_base_df(path: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(path, CHECKPOINT_DF_FNAME))


@task(name='clean_data', log_stdout=True)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    log = prefect.context.get('logger')
    before = len(df)
    df.dropna(inplace=True)
    log.info('dropped na changing items from %s -> %s', before, len(df))
    # create binary negativity variable
    df['negativity'] = df.Sentiment.apply(lambda x: 1 if x.lower() == 'negative' else 0)
    return df


@task(name='extract_features', log_stdout=True)
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    log = prefect.context.get('logger')
    # TODO use config with kwargs?
    audio_limit = 5 * 22050  # 5 sec * sr
    mel_window_length = 8
    mfcc_window_length = 512
    n_mfcc = 20
    chroma_window_length = 32
    n_chroma = 12
    features = []
    for i, r in df.iterrows():
        if not i % 500:
            log.info('Extract progress: %s / %s', i, len(df))

        audio = r.audio[:audio_limit]
        features.append(
            np.hstack((
                extract_melspectrogram(audio, r.sr, mel_window_length),
                extract_mfcc(audio, r.sr, mfcc_window_length, n_mfcc),
                extract_chroma(audio, r.sr, chroma_window_length, n_chroma)
            ))
        )

    # add features into original df
    return df.combine_first(pd.DataFrame(np.array(features)))


@task(name='run_bayes_search', log_stdout=True)
def run_bayes_search(df: pd.DataFrame) -> None:
    # TODO: consider adding a CL switch to run hyperparameter search
    return None


@task(name='train_model', log_stdout=True)
def train_test_model(df: pd.DataFrame) -> Tuple[BaseEstimator, dict]:
    log = prefect.context.get('logger')
    test_size = 0.2
    X = get_features(df)
    y = df.negativity
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=0)
    log.info('Train set items: %s, Test set items: %s', len(y_train), len(y_test))

    pipe = get_ml_pipeline()
    pipe.fit(X_train, y_train)
    train_score = f1_score(y_train, pipe.predict(X_train), average='weighted')
    test_score = f1_score(y_test, pipe.predict(X_test), average='weighted')
    log.info('Train f1 score %s, Test f1 score %s', train_score, test_score)
    metrics = {'f1_score': {'train': train_score, 'test': test_score}}
    return pipe['estimator'], metrics


def get_ml_pipeline() -> Pipeline:
    pca_components = 80
    return Pipeline([
        ('standardize', QuantileTransformer(output_distribution='normal', random_state=0)),
        ('decomposition', PCA(random_state=0, n_components=pca_components)),
        ('oversample', ADASYN(random_state=0, n_jobs=-1)),
        ('estimator', get_baked_model()),
    ])


@task(name='evaluate_parallel_prediction', log_stdout=True)
def eval_parallel_pred(model, df):
    log = prefect.context.get('logger')
    duplication = 3
    X = get_features(df)
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


def dupe_data(X, duplication: int):
    return da.concatenate([da.from_array(X.to_numpy(), chunks=X.shape)
                           for _ in range(duplication)])


@task(name='record_results', log_stdout=True)
def record_results(data_dir: str, model: BaseEstimator, dag_config: dict, metrics: dict) -> None:
    # save config / hyperparameters
    # save metrics to disk
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


def get_flow() -> Flow:
    with Flow('sentiment_classifier_ETL') as flow:
        # specify ETL DAG to prefect
        data_dir = Parameter('data_dir', default=DATA_DIR)
        df_dir = Parameter('df_dir', default=DF_DIR)
        # TODO pass dag_config into tasks and use it
        dag_config = Parameter('dag_config', default=DAG_CONFIG)

        df_exists = base_df_exists(df_dir)
        with case(df_exists, False):
            wavs_dirs = extract_wavs(data_dir)
            labels_dfs = load_data_labels(data_dir)
            df_true = join_audio_to_labels(wavs_dirs, labels_dfs)
            df_true = extract_features(df_true)

            # save time-consuming work
            checkpoint_df(df_true, df_dir)

        with case(df_exists, True):
            df_false = load_base_df(df_dir)

        # join workflows
        df = merge(df_true, df_false)
        df = clean_data(df)
        model, metrics = train_test_model(df)
        eval_parallel_pred(model, df)
        record_results(data_dir, model, dag_config, metrics)

    return flow


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    return parser.parse_args()


def run_pipeline():
    args = get_args()

    # Prefect tasks make use of libraries like numpy, pandas, or scikit-learn that release the GIL
    executor = LocalDaskExecutor(scheduler='threads')
    flow = get_flow()
    flow.run(executor=executor)


if __name__ == '__main__':
    run_pipeline()
