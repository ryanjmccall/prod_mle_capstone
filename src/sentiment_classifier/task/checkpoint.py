import os
import sqlite3

import numpy as np
import pandas as pd
import prefect
from prefect import task


_CHECKPOINT_DF_FNAME = 'checkpoint.sqlite'
_SQLITE_CHECKPOINT_TABLE_NAME = 'df_checkpoint'
_SQLITE_CHECKPOINT_INDEX_LABEL = 'index'


@task(name='write_checkpoint', log_stdout=True)
def write_checkpoint(df: pd.DataFrame, path: str):
    """Saves Dataframe to sqlite table at given path.

    :param df: Dataframe containing processed features
    :param path: directory where checkpoint will be written
    """
    logger = prefect.context.get('logger')
    if not os.path.exists(path):
        os.makedirs(path)

    logger.info('Saving feature df to SQLite table at %s', path)
    con = _sqlite_conn(path)
    df.to_sql(_SQLITE_CHECKPOINT_TABLE_NAME, con, if_exists='replace', index=True,
              index_label=_SQLITE_CHECKPOINT_INDEX_LABEL)


def _sqlite_conn(checkpoint_dir: str):
    # This could be replaced with a connection to a database server
    sql_path = os.path.join(checkpoint_dir, 'checkpoint.sqlite')
    return sqlite3.connect(sql_path)


@task(name='checkpoint_exists', log_stdout=True)
def checkpoint_exists(path: str) -> bool:
    """Returns whether feature df checkpoint is available.

    :param path: checkpoint directory
    :return: True if a checkpoint exists at path
    """
    logger = prefect.context.get('logger')
    exists = os.path.exists(os.path.join(path, _CHECKPOINT_DF_FNAME))
    logger.info('Feature df checkpoint available: %s', exists)
    return exists


@task(name='load_checkpoint')
def load_checkpoint(path: str) -> pd.DataFrame:
    """Loads feature-df checkpoint from SQLite table.

    :param path: checkpoint directory
    :return: Loaded dataframe
    """
    con = _sqlite_conn(path)
    qry = 'SELECT * FROM ' + _SQLITE_CHECKPOINT_TABLE_NAME
    df = pd.read_sql(qry, con, index_col=_SQLITE_CHECKPOINT_INDEX_LABEL)
    df['features'] = df['features'].apply(func=_bytes_to_np)
    return df


def _bytes_to_np(x: bytes) -> np.ndarray:
    return np.frombuffer(x, dtype=np.float32)
