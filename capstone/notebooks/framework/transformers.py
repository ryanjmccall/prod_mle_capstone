import librosa
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractTransformer(BaseEstimator, TransformerMixin):
    """Trims signal removing all data beyond given limit, then extracts specified features."""

    def __init__(self, sr: int, limit: int = None, mel=True, mfcc=True, n_mfcc: int = 20, chroma=True):
        self.sr = sr
        self.limit = limit
        self.mel = mel
        self.mfcc = mfcc
        self.n_mfcc = n_mfcc
        self.chroma = chroma

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = []
        #         for r in tqdm_notebook(X, desc=f'Extract with limit={limit}: '):
        for r in X:
            row = r[:self.limit] if self.limit else r
            feature_groups = []
            if self.mel:
                feature_groups.append(self.extract_melspectrogram(row))
            if self.mfcc:
                feature_groups.append(self.extract_mfcc(row))
            if self.chroma:
                feature_groups.append(self.extract_chroma(row))
            result.append(np.hstack(feature_groups))

        return np.array(result)

    def extract_melspectrogram(self, X):
        return np.mean(librosa.feature.melspectrogram(X, sr=self.sr).T, axis=0)

    def extract_mfcc(self, X):
        return np.mean(librosa.feature.mfcc(y=X, sr=self.sr, n_mfcc=self.n_mfcc).T, axis=0)

    def extract_chroma(self, X):
        stft = np.abs(librosa.stft(X))
        return np.mean(librosa.feature.chroma_stft(S=stft, sr=self.sr).T, axis=0)
