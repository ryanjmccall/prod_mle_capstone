import os
import pandas as pd
import shutil
import unittest

from sentiment_classifier.context import DATA_DIR
from sentiment_classifier.task.checkpoint import (CHECKPOINT_DF_FNAME, checkpoint_exists, load_checkpoint,
                                                  write_checkpoint)


class TestCheckpoint(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.DataFrame([1, 2, 3])
        self.checkpoint_dir = os.path.join(DATA_DIR, 'testing')
        self.checkpoint_file = os.path.join(self.checkpoint_dir, CHECKPOINT_DF_FNAME)

    def tearDown(self) -> None:
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

    def test_write_checkpoint(self):
        write_checkpoint.run(self.df, self.checkpoint_dir)

        assert os.path.exists(self.checkpoint_file)

    def test_checkpoint_exists_false(self):
        assert not checkpoint_exists.run(self.checkpoint_dir)

    def test_checkpoint_exists_true(self):
        write_checkpoint.run(self.df, self.checkpoint_dir)

        assert checkpoint_exists.run(self.checkpoint_dir)

    def test_load_checkpoint(self):
        write_checkpoint.run(self.df, self.checkpoint_dir)

        result = load_checkpoint.run(self.checkpoint_dir)

        assert self.df.equals(result)

    # TODO error cases
