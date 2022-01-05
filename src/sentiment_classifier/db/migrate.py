import numpy as np
import psycopg2
from sqlalchemy import create_engine

from sentiment_classifier.context import DF_DIR
from sentiment_classifier.task.checkpoint import load_checkpoint


def get_psycopg_conn():
    # manually create the user and password by hand and set in OS env vars
    host = 'localhost'
    dbname = 'postgres'
    user = 'newuser'
    password = 'password'
    return psycopg2.connect(
        host=host,
        database=dbname,
        user=user,
        password=password
    )


def get_sqlalchemy_conn():
    host = 'localhost'
    database = 'postgres'
    user = 'newuser'
    password = 'password'
    conn_str = f'postgresql://{user}:{password}@{host}/{database}'
    return create_engine(conn_str)


def main():
    engine = get_sqlalchemy_conn()

    print('load df')
    df = load_checkpoint.run(DF_DIR)
    DROP_COLS = ['Sr No.', 'Utterance', 'Speaker', 'Emotion', 'Sentiment', 'Dialogue_ID', 'Utterance_ID',
                 'Season', 'Episode', 'StartTime', 'EndTime']
    df.drop(labels=DROP_COLS, axis=1, inplace=True)

    for col in ('features', 'audio'):
        df[col] = df[col].apply(lambda r: r.tobytes())
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

# TODO Look into this trick:
# https://towardsdatascience.com/upload-your-pandas-dataframe-to-your-database-10x-faster-eb6dc6609ddf
"""
#perform to_sql test and print result
db = create_engine(conn_string)
conn = db.connect()

start_time = time.time()
df.to_sql('to_sql_test', con=conn, if_exists='replace', index=False)
print("to_sql duration: {} seconds".format(time.time() - start_time))



#perform COPY test and print result
sql = '''
COPY copy_test
FROM 'PATH_TO_FILE.csv' --input full file path here. see line 46
DELIMITER ',' CSV;
'''

table_create_sql = '''
CREATE TABLE IF NOT EXISTS copy_test (id                bigint,
                                      quantity          int,
                                      cost              double precision,
                                      total_revenue     double precision)
'''

pg_conn = psycopg2.connect(conn_string)
cur = pg_conn.cursor()
cur.execute(table_create_sql)
cur.execute('TRUNCATE TABLE copy_test') #Truncate the table in case you've already run the script before

start_time = time.time()
df.to_csv('upload_test_data_from_copy.csv', index=False, header=False) #Name the .csv file reference in line 29 here
cur.execute(sql)
pg_conn.commit()
cur.close()
print("COPY duration: {} seconds".format(time.time() - start_time))



#close connection
conn.close()
"""


if __name__ == '__main__':
    main()
