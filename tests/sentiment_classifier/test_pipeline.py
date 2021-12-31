import sentiment_classifier.pipeline as pipe


class TestPipeline(object):

    def test_unique_fname_standard(self):
        name = '/a/b/0_1.wav'

        res = pipe.unique_fname(name)

        assert res == '0_1'

    def test_unique_fname_diautt(self):
        name = '/a/b/dia0_utt1.wav'

        res = pipe.unique_fname(name)

        assert res == '0_1'
