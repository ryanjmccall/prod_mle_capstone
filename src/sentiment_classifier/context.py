import os


#: sentiment_classifier package root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


#: MELD dataset directory
DATA_DIR = os.path.join(ROOT_DIR, '../../data')


#: features checkpoint directory
DF_DIR = os.path.join(DATA_DIR, 'features')


LOG_DIR = os.path.join(ROOT_DIR, '../../logs')
