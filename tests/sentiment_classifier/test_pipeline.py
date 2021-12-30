import sentiment_classifier.pipeline as pipe


class TestPipeline(object):

    def test_unique_fname(self):
        name = '/a/b/0_1.wav'

        res = pipe.unique_fname(name)

        assert res == '0_1'
