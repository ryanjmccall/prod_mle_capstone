import pandas as pd
import prefect
from prefect import task

from sentiment_classifier.task.extract_helper import extract_features


@task(name='extract_features', log_stdout=True)
def extract_features_task(df: pd.DataFrame, dag_conf: dict) -> pd.DataFrame:
    """Prefect task to extract librosa features from MELD audio.

    :param df: MELD dataset audio
    :param dag_conf: ETL dag configuration containing extract parameters
    :return: dataset Dataframe with extracted features added under new 'features' column as numpy array
    """
    conf = dag_conf['extract']
    audio_limit = conf['audio_limit']
    mel_window_length = conf['mel_window_length']
    n_mels = conf['n_mels']
    mfcc_window_length = conf['mfcc_window_length']
    n_mfcc = conf['n_mfcc']
    chroma_window_length = conf['chroma_window_length']
    n_chroma = conf['n_chroma']

    def extract_helper(r):
        """Helper to extract features from df row"""
        if r.audio is None or r.sr is None:
            raise ValueError('Row audio/sr cannot be None')

        return extract_features(r.audio, r.sr, audio_limit,
                                mel_window_length, n_mels,
                                mfcc_window_length, n_mfcc,
                                chroma_window_length, n_chroma)

    logger = prefect.context.get('logger')
    logger.info('Extracting features to "features" df column')
    df['features'] = df.apply(extract_helper, axis=1)
    return df
