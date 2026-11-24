import unittest

from plot import eligible_series, plot_group, workload


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

    def test_short_sweeps_omitted(self):
        def rows(ns):
            return [dict(length=n, case_id=str(n)) for n in ns]
        series = {"a": rows([1, 2, 3, 4]), "b": rows([1, 2, 3, 4]),
                  "short": rows([1, 2, 3]), "disjoint": rows([5, 6, 7, 8])}
        self.assertEqual(set(eligible_series(series)), {"a", "b"})
        self.assertFalse(eligible_series({"a": rows([1, 2, 3, 4])}))

    def test_combined_rank_one_sweep(self):
        self.assertEqual(plot_group("1d"), plot_group("long"))
        self.assertNotEqual(plot_group("batched"), plot_group("1d"))
