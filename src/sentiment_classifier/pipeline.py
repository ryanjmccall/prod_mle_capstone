from copy import deepcopy

import lightgbm as lgb
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


def get_train_pipeline(dag_config: dict) -> ImbalancedPipeline:
    """Get predetermined ML training pipeline.

    :param dag_config: DAG/ETL configuration
    :return: training Pipeline
    """
    return ImbalancedPipeline([
        ('standardize', QuantileTransformer(**dag_config['standardize'])),
        ('decomposition', PCA(**dag_config['decomposition'])),
        ('oversample', ADASYN(**dag_config['oversample'])),
        ('model', lgb.LGBMClassifier(**dag_config['model'])),
    ])


def get_prediction_pipeline(pipe) -> Pipeline:
    """Gets the production prediction pipeline without oversampling step.

    :param: training pipeline
    :return: new prediction pipeline
    """
    copy = deepcopy(pipe)
    del copy.steps[2]
    return Pipeline(steps=copy.steps)
