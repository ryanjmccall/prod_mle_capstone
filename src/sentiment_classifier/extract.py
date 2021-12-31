import librosa
import numpy as np


def extract_melspectrogram(X, sr, win_length):
    hop_length = int(win_length / 4)
    mel = librosa.feature.melspectrogram(
        y=X,
        sr=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length
    )
    return np.mean(mel.T, axis=0)


def extract_mfcc(X, sr, win_length, n_mfcc):
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