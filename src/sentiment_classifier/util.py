import os
from typing import Generator


def unique_fname(full_path: str) -> str:
    """Get unique file name for given full path to MELD data file. The return format is '[dialog]_[utterance]'.

    :param full_path: full path to MELD .mp4 data file
    :return: unique id of data file (only unique within dataset directory)
    """
    fname = full_path.split('/')[-1].split('.')[0]
    return fname.replace('dia', '').replace('utt', '').replace('final_videos_test', '')


def listdir_ext(path: str, extension: str) -> Generator[str, None, None]:
    """List all files in path having extension.

    :param path directory path
    :param extension: file extension
    :return: generator on files at path having extension
    """
    return (f for f in os.listdir(path) if f.split('.')[-1] == extension)
