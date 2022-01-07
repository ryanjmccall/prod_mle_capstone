import os

import pandas as pd
import prefect
from prefect import task


CHECKPOINT_DF_FNAME = 'df_checkpoint.pkl'


@task(name='checkpoint_df', log_stdout=True)
def write_checkpoint(df: pd.DataFrame, path: str):
    logger = prefect.context.get('logger')
    if not os.path.exists(path):
        os.makedirs(path)

    out_path = os.path.join(path, CHECKPOINT_DF_FNAME)
    logger.info('Saving df to %s', out_path)
    df.to_pickle(out_path)


@task(name='df_checkpoint_exists', log_stdout=True)
def checkpoint_exists(path: str) -> bool:
    logger = prefect.context.get('logger')
    exists = os.path.exists(os.path.join(path, CHECKPOINT_DF_FNAME))
    logger.info('Base df available: %s', exists)
    return exists


@task(name='load_base_df')
def load_checkpoint(path: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(path, CHECKPOINT_DF_FNAME))
