import json
import os
from datetime import datetime

import prefect
from imblearn.pipeline import Pipeline
from joblib import dump
from prefect import task

from sentiment_classifier.pipeline import get_predict_pipeline


@task(name='record_results', log_stdout=True)
def record_results(data_dir: str, train_pipe: Pipeline, dag_config: dict, metadata: dict) -> str:
    """Record the pipeline, config, and metadata from a DAG run.

    Converts the training pipeline to a prediction pipeline ready for production.
    """
    # TODO consider using MLFlow OS library (https://mlflow.org/) to record "ML lifecycle"
    logger = prefect.context.get('logger')

    dt = str(datetime.now())
    results_path = os.path.join(data_dir, 'results', dt)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    logger.info('Recording DAG run results to: %s', results_path)

    predict_pipe = get_predict_pipeline(train_pipe)
    pipe_path = os.path.join(results_path, 'prediction_pipeline.joblib')
    dump(predict_pipe, filename=pipe_path)
    logger.info('Prediction pipeline saved to: %s', pipe_path)

    # TODO convert the skopt space objects to lists, for now just omit them
    dag_config['bayes_search']['search_spaces'] = None
    record = dict(dag_config=dag_config, metadata=metadata)
    record_path = os.path.join(results_path, 'record.json')
    with open(record_path, 'w', encoding='utf8') as f:
        json.dump(record, f)

    logger.info('DAG run records saved to %s', record_path)
    return results_path
