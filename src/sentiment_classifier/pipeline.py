import argparse
import logging
import os
import sys
from typing import List, Generator

import librosa
import moviepy.editor as mp
import pandas as pd
import prefect
from prefect import task, Flow


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, '../../data')
LOG_DIR = os.path.join(ROOT_DIR, '../../logs')


log = logging.getLogger(__name__)


def unique_fname(full_path: str) -> str:
    """Standardizes MELD data file name to 'dialog_utterance' form"""
    fname = full_path.split('/')[-1].split('.')[0]
    return fname.replace('dia', '').replace('utt', '').replace('final_videos_test', '')


def listdir_ext(path: str, extension: str) -> Generator[str, None, None]:
    return (f for f in os.listdir(path) if f.split('.')[-1] == extension)


@task(name='extract_wavs')
def extract_wavs() -> List[str]:
    """The MELD dataset provides 3 dirs of .mp4 files. This function converts all .mp4 files
    to .wav and returns the .wav dir. If a .wav file already exists,
    the conversion is skipped.
    """
    exceptions = []
    converted = 0
    for dir_ in ('train', 'dev', 'test'):
        # N.B. we have duplicate dia_utt ids among the 3 folders, so we have to operate separately
        # until we have all 3 dataframes connecting audio and labels
        dest_path = os.path.join(DATA_DIR, 'audio', dir_)
        finished_fnames = set(f.split('.')[0] for f in listdir_ext(dest_path, 'wav'))
        log.info('Found %s converted .wav files in %s', len(finished_fnames), dest_path)

        src_path = os.path.join(DATA_DIR, 'raw', dir_)
        mp4_paths = (os.path.join(src_path, f) for f in listdir_ext(src_path, 'mp4'))
        for mp4_path in mp4_paths:
            fname = unique_fname(mp4_path)
            if fname in finished_fnames:
                continue

            out_path = os.path.join(DATA_DIR, 'audio', dir_, fname + '.wav')
            try:
                clip = mp.VideoFileClip(mp4_path)

                # highest quality kwargs based on https://zulko.github.io/moviepy/ref/AudioClip.html
                clip.audio.write_audiofile(out_path, nbytes=4, codec='pcm_s32le', bitrate='3000k')
                converted += 1
            except Exception as e:
                # one dataset file was found to be corrupt
                exceptions.append(e)

    log.info('Converted %s files and had %s exceptions(s): \n%s', converted, len(exceptions), exceptions)
    return [os.path.join(DATA_DIR, 'audio', dir_) for dir_ in ('train', 'dev', 'test')]


# TODO check df.isna().values.any()
@task(name='load_data_labels')
def load_data_labels() -> List[pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'labels', 'train_labels.csv'))
    dev_df = pd.read_csv(os.path.join(DATA_DIR, 'labels', 'dev_labels.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'labels', 'test_labels.csv'))

    # construct unique ID
    cols = ['Dialogue_ID', 'Utterance_ID']
    train_df['dia_utt'] = train_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    dev_df['dia_utt'] = dev_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    test_df['dia_utt'] = test_df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return [train_df, dev_df, test_df]


@task(name='join_audio_to_labels')
def join_audio_to_labels(wav_dirs: str, label_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """For each row in labels DataFrame, load corresponding audio and add it to the DataFrame."""
    dfs = []
    for wav_dir, df in zip(wav_dirs, label_dfs):
        audios, srs = [], []
        for uid in df['dia_utt']:
            path = os.path.join(wav_dir, uid + '.wav')
            if os.path.exists(path):
                audio, sr = librosa.load(path)
            else:
                audio = None
                sr = None
                log.warning('Missing wav: %s', path)

            audios.append(audio)
            srs.append(sr)

        df['audio'] = audios
        df['sr'] = srs
        dfs.append(df)

    full_df = pd.concat(dfs)
    before = len(full_df)
    full_df.dropna(inplace=True)
    log.info('Joined audio to labels and dropped na %s -> %s', before, len(full_df))
    return full_df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='increase output verbosity', action='store_true')
    return parser.parse_args()


def run_pipeline():
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s | %(message)s',
        # filename=os.path.join(LOG_DIR, 'pipeline.log'),
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'pipeline.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    if args.verbose:
        log.info('verbose on')

    with Flow('sentiment_classifier_ETL') as flow:
        # specify ETL execution DAG to prefect
        wavs_dirs = extract_wavs()
        labels_dfs = load_data_labels()
        dataset_df = join_audio_to_labels(wavs_dirs, labels_dfs)


    flow.run()


if __name__ == '__main__':
    run_pipeline()
