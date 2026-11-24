import unittest

from summarize import aggregate, geomean


class TestSummary(unittest.TestCase):
    def test_ratio_direction(self):
        self.assertAlmostEqual(geomean([0.25, 1.0]), 0.5)

    def test_equal_runner_weight(self):
        row = dict(cpu="CPU", api="Python", level="v3", package="ducc-py",
                   candidate="ducc-py-v3", revision="sha", group="1d",
                   reference_mode="alloc", timing_mode="alloc")
        rows = [dict(row, runner="a", ratio=0.25)] * 4 + [dict(row, runner="b", ratio=1)]
        result = aggregate(rows, "isa", "CPU", by_isa=True)[0]
        self.assertAlmostEqual(result["ratio"], 0.5)
        self.assertEqual(result["cases"], 5)
        self.assertEqual(result["runners"], 2)

    def test_revisions_not_mixed(self):
        row = dict(cpu="CPU", api="Python", level="v3", package="ducc-py",
                   candidate="ducc-py-v3", group="1d", reference_mode="alloc",
                   timing_mode="alloc", runner="a", ratio=0.5)
        self.assertEqual(len(aggregate([dict(row, revision="a"), dict(row, revision="b")],
                                       "cpu", "CPU")), 2)
