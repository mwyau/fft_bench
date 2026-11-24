import unittest

from plot import workload


class TestPlot(unittest.TestCase):
    def test_transformed_length(self):
        row = dict(kind="fft", precision="f64", layout="C", threads=1,
                   shape=[8, 32, 64], axes=[-2, -1], lengths=[],
                   mode="default", norm="backward", transform_type=2)
        length, panel, lengths = workload(row)
        self.assertEqual(length, 2048)
        self.assertEqual(lengths, (32, 64))
        self.assertEqual(panel[5], ((0, 8),))

    def test_resize_length(self):
        row = dict(kind="fft", precision="f64", layout="C", threads=1,
                   shape=[1024], axes=[0], lengths=[1031],
                   mode="default", norm="backward", transform_type=2)
        self.assertEqual(workload(row)[0], 1031)
