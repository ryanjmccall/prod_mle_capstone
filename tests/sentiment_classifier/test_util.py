import unittest

from sentiment_classifier.context import ROOT_DIR
from sentiment_classifier.util import unique_fname, listdir_ext


class TestUtil(unittest.TestCase):

    def test_unique_fname_standard(self):
        name = '/a/b/0_1.wav'

        fname = unique_fname(name)

        assert fname == '0_1'

    def test_unique_fname_diautt(self):
        name = '/a/b/dia0_utt1.wav'

        fname = unique_fname(name)

        assert fname == '0_1'

    def test_listdir_ext_txt(self):
        assert not list(listdir_ext(ROOT_DIR, 'txt'))

    def test_listdir_ext_py(self):
        assert list(listdir_ext(ROOT_DIR, 'py'))

    # TODO corner cases?