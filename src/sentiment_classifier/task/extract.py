import librosa
import numpy as np
import pandas as pd
from prefect import task

from sentiment_classifier.config.default_config import *


@task(name='extract_features')
def extract_features_task(df: pd.DataFrame, dag_conf: dict) -> pd.DataFrame:
    """Extract librosa features from audio based on supplied config."""
    conf = dag_conf['extract']
    audio_limit = conf['audio_limit']
    mel_window_length = conf['mel_window_length']
    n_mels = conf['n_mels']
    mfcc_window_length = conf['mfcc_window_length']
    n_mfcc = conf['n_mfcc']
    chroma_window_length = conf['chroma_window_length']
    n_chroma = conf['n_chroma']

    def extract_helper(r):
        if r.audio is None or r.sr is None:
            raise ValueError('Row audio/sr cannot be None')

        return extract_features(r.audio, r.sr, audio_limit,
                                mel_window_length, n_mels,
                                mfcc_window_length, n_mfcc,
                                chroma_window_length, n_chroma)

    df['features'] = df.apply(extract_helper, axis=1)
    return df


def extract_features(
    audio,
    sr,
    audio_limit=AUDIO_LIMIT,
    mel_window_length=MEL_WINDOW_LENGTH,
    n_mels=N_MELS,
    mfcc_window_length=MFCC_WINDOW_LENGTH,
    n_mfcc=N_MFCC,
    chroma_window_length=CHROMA_WINDOW_LENGTH,
    n_chroma=N_CHROMA
):
    audio = audio[:int(audio_limit)]
    return np.hstack((
        extract_melspectrogram(audio, sr, mel_window_length, n_mels),
        extract_mfcc(audio, sr, mfcc_window_length, n_mfcc),
        extract_chroma(audio, sr, chroma_window_length, n_chroma)
    ))


def extract_melspectrogram(X, sr, win_length, n_mels):
    """Extract melspectrogram from given audio signal and return average across time
    for each mel-frequency domain."""
    hop_length = int(win_length / 4)
    mel = librosa.feature.melspectrogram(
        y=X,
        sr=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # mel shape is (n_mels, windows)
    # mel.T is (windows, n_mels), then average across rows
    return np.mean(mel.T, axis=0)


def extract_mfcc(X, sr, win_length, n_mfcc):
    """Extract mfcc coefficients from given audio signal and return average across time
    for each coefficient."""
    hop_length = int(win_length / 4)
    mfcc = librosa.feature.mfcc(
        y=X,
        sr=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        n_mfcc=n_mfcc
    )
    return np.mean(mfcc.T, axis=0)


def extract_chroma(X, sr, win_length, n_chroma):
    """Extract chroma from given audio signal and return average across time
    for each chroma."""
    hop_length = int(win_length / 4)
    chroma = librosa.feature.chroma_stft(
        y=X,
        sr=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        n_chroma=n_chroma
    )
    return np.mean(chroma.T, axis=0)
