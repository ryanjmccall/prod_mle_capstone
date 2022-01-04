import os
from typing import List

import librosa
from moviepy import editor as mp
import pandas as pd
import prefect
from prefect import task

from sentiment_classifier.util import listdir_ext, unique_fname


@task(name='convert_wavs', log_stdout=True)
def convert_mp4_to_wav(data_dir: str) -> List[str]:
    """The MELD dataset provides 3 dirs of .mp4 files. This function converts all .mp4 files
    to .wav and returns the .wav dir. If a .wav file already exists,
    the conversion is skipped.

    Expects the following input directory structure:
    - data_dir/raw/[train|dev|test]
    - data_dir/labels/[train|dev|test]

    Because of file namespace collisions, outputs wavs to following structure:
    - data_dir/audio/[train|dev|test]
    """
    log = prefect.context.get('logger')
    dest_paths = []
    exceptions = []
    converted = 0
    for dir_ in ('train', 'dev', 'test'):
        # N.B. we have duplicate dia_utt ids among the 3 folders, so we have to operate separately
        # until we have all 3 dataframes connecting audio and labels
        dest_path = os.path.join(data_dir, 'audio', dir_)
        dest_paths.append(dest_path)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        finished_fnames = set(f.split('.')[0] for f in listdir_ext(dest_path, 'wav'))
        log.info('Found %s converted .wav files in %s', len(finished_fnames), dest_path)

        src_path = os.path.join(data_dir, 'raw', dir_)
        mp4_paths = (os.path.join(src_path, f) for f in listdir_ext(src_path, 'mp4'))
        for mp4_path in mp4_paths:
            fname = unique_fname(mp4_path)
            if fname in finished_fnames:
                continue

            out_path = os.path.join(data_dir, 'audio', dir_, fname + '.wav')
            try:
                clip = mp.VideoFileClip(mp4_path)

                # highest quality kwargs based on https://zulko.github.io/moviepy/ref/AudioClip.html
                clip.audio.write_audiofile(out_path, nbytes=4, codec='pcm_s32le', bitrate='3000k')
                converted += 1
            except Exception as e:
                # one dataset file was found to be corrupt
                exceptions.append(e)

    log.info('Converted %s files and had %s exceptions(s): \n%s', converted, len(exceptions), exceptions)
    return dest_paths


@task(name='load_data_labels')
def load_labels(data_dir: str) -> List[pd.DataFrame]:
    dfs = [pd.read_csv(os.path.join(data_dir, 'labels', file))
           for file in ('train_labels.csv', 'dev_labels.csv', 'test_labels.csv')]

    cols = ['Dialogue_ID', 'Utterance_ID']
    for df in dfs:
        # construct unique ID
        df['dia_utt'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

    return dfs


@task(name='add_audio_to_labels', log_stdout=True)
def add_audio_to_labels(wav_dirs: str, label_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """For each row in labels DataFrame, load corresponding audio and add it to the DataFrame."""
    log = prefect.context.get('logger')
    dfs = []
    for wav_dir, df in zip(wav_dirs, label_dfs):
        log.info('Joining audio from %s to labels df of size %s', wav_dir, len(df))
        audios, srs = [], []
        for uid in df['dia_utt']:
            path = os.path.join(wav_dir, uid + '.wav')
            if os.path.exists(path):
                log.debug('Loading wav %s', path)
                audio, sr = librosa.load(path)
            else:
                log.warning('Wav not found at: %s', path)
                audio = None
                sr = None

            audios.append(audio)
            srs.append(sr)

        df['audio'] = audios
        df['sr'] = srs
        dfs.append(df)

    return pd.concat(dfs)
