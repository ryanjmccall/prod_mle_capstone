from copy import deepcopy

import lightgbm as lgb
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


def get_train_pipeline(conf: dict) -> ImbalancedPipeline:
    """Get predetermined ML training pipeline."""
    return ImbalancedPipeline([
        ('standardize', QuantileTransformer(**conf['standardize'])),
        ('decomposition', PCA(**conf['decomposition'])),
        ('oversample', ADASYN(**conf['oversample'])),
        ('model', lgb.LGBMClassifier(**conf['model'])),
    ])


def get_prediction_pipeline(pipe) -> Pipeline:
    """Gets the production prediction pipeline without oversampling"""
    copy = deepcopy(pipe)
    del copy.steps[2]
    return Pipeline(steps=copy.steps)
