import os
from typing import Generator


def unique_fname(full_path: str) -> str:
    """Get unique file name for given full path to MELD data file. The return format is '[dialog]_[utterance]'."""
    fname = full_path.split('/')[-1].split('.')[0]
    return fname.replace('dia', '').replace('utt', '').replace('final_videos_test', '')


def listdir_ext(path: str, extension: str) -> Generator[str, None, None]:
    return (f for f in os.listdir(path) if f.split('.')[-1] == extension)
