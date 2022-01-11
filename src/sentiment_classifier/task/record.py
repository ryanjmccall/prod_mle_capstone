import json
import os
from datetime import datetime

import prefect
from imblearn.pipeline import Pipeline
from joblib import dump
from prefect import task

from sentiment_classifier.pipeline import get_prediction_pipeline


@task(name='record_results', log_stdout=True)
def record_results(data_dir: str, train_pipe: Pipeline, dag_config: dict, metadata: dict) -> str:
    """Record the result of running the ETL training pipeline.

    Converts the training sklearn pipeline to a prediction pipeline ready for production and saves it to
    model store (here just disk).

    :param data_dir: parent directory of the results
    :param train_pipe: the sklearn training Pipeline used in this ETL run
    :param dag_config: the configuration for used this ETL run
    :param metadata: additional data generated during the ETL run to be recorded
    :return: path to new directory containing the full record of this ETL run
    """
    # TODO consider using MLFlow OS library (https://mlflow.org/) to record "ML lifecycle"
    logger = prefect.context.get('logger')

    dt = str(datetime.now())
    results_path = os.path.join(data_dir, 'results', dt)
    logger.info('Recording DAG run results to: %s', results_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Get prediction pipeline and serialize it to disk
    pred_pipe = get_prediction_pipeline(train_pipe)
    pipe_path = os.path.join(results_path, 'prediction_pipeline.joblib')
    dump(pred_pipe, filename=pipe_path)
    logger.info('Prediction pipeline prefix saved to: %s', pipe_path)

    # TODO convert the skopt space objects to lists, for now just omit them
    dag_config['bayes_search']['search_spaces'] = None
    record = dict(dag_config=dag_config, metadata=metadata)
    record_path = os.path.join(results_path, 'record.json')
    with open(record_path, 'w', encoding='utf8') as f:
        json.dump(record, f)

    logger.info('DAG run records saved to %s', record_path)
    return results_path
