from copy import deepcopy

import lightgbm as lgb
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


def get_train_pipeline(conf: dict) -> Pipeline:
    return Pipeline([
        ('standardize', QuantileTransformer(**conf['standardize'])),
        ('decomposition', PCA(**conf['decomposition'])),
        ('oversample', ADASYN(**conf['oversample'])),
        ('model', lgb.LGBMClassifier(**conf['model'])),
    ])


def get_predict_pipeline(train_pipeline) -> Pipeline:
    predict_pipeline = deepcopy(train_pipeline)
    if len(predict_pipeline.steps) < 3:
        raise ValueError('Unexpected pipeline size')
    step_name, _ = predict_pipeline.steps[2]
    if step_name != 'oversample':
        raise ValueError('Expected oversample step but was %s', step_name)

    del predict_pipeline.steps[2]  # don't want oversampling for prediction
    return predict_pipeline