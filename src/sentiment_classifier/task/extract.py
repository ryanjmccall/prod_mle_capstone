import librosa
import numpy as np
import pandas as pd
from prefect import task


@task(name='extract_features')
def extract_features(df: pd.DataFrame, conf: dict) -> pd.DataFrame:
    """Extract librosa features from audio based on supplied config."""
    audio_limit = conf['audio_limit']
    mel_window_length = conf['mel_window_length']
    n_mels = conf['n_mels']
    mfcc_window_length = conf['mfcc_window_length']
    n_mfcc = conf['n_mfcc']
    chroma_window_length = conf['chroma_window_length']
    n_chroma = conf['n_chroma']

    def extract_helper(r):
        audio = r.audio[:audio_limit]
        return np.hstack((
            extract_melspectrogram(audio, r.sr, mel_window_length, n_mels),
            extract_mfcc(audio, r.sr, mfcc_window_length, n_mfcc),
            extract_chroma(audio, r.sr, chroma_window_length, n_chroma)
        ))

    df['features'] = df.apply(extract_helper, axis=1)
    return df


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
