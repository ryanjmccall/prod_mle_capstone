import os

import pandas as pd
import psycopg2
from sqlalchemy import create_engine

from sentiment_classifier.context import DF_DIR
from sentiment_classifier.task.checkpoint import load_checkpoint


def get_db_creds() -> dict:
    return dict(host=os.environ.get('CLASSIFIER_FEATURES_HOST'),
                database=os.environ.get('CLASSIFIER_FEATURES_DATABASE'),
                user=os.environ.get('CLASSIFIER_FEATURES_USER'),
                password=os.environ.get('CLASSIFIER_FEATURES_PASSWORD'))


def get_psycopg_conn():
    creds = get_db_creds()
    return psycopg2.connect(**creds)


def get_sqlalchemy_conn():
    creds = get_db_creds()
    conn_str = 'postgresql://{user}:{password}@{host}/{database}'.format(**creds)
    return create_engine(conn_str)


def save_features_to_db(df: pd.DataFrame) -> None:
    engine = get_sqlalchemy_conn()




    df['features'] = df['features'].apply(lambda r: r.tobytes())
    # inverse operation: np.frombuffer(y, dtype=np.float32)

    # TODO look into the adapter registry to tell sqlalchemy to do this conversion? or above might suffice

    print('to sql')
    df.to_sql(name='meld_features', con=engine, if_exists='replace')

    res = engine.execute("SELECT * FROM meld_features LIMIT 1").fetchall()
    print(res)

    # cur.execute('SELECT * FROM test')
    # print(cur.fetchall())
    #
    # conn.commit()
    #
    # cur.close()
    # conn.close()


def load_features_from_db() -> pd.DataFrame:
    pass


if __name__ == '__main__':
    save_features_to_db(df=load_checkpoint.run(DF_DIR))

