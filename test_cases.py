import unittest

from cases import cases


class TestCases(unittest.TestCase):
    def test_fast_subset(self):
        fast = {c.id for c in cases("fast")}
        full = {c.id for c in cases("full")}
        self.assertLess(fast, full)

    def test_unique_ids(self):
        for suite in ("fast", "full"):
            rows = cases(suite)
            self.assertEqual(len(rows), len({c.id for c in rows}))

    def test_no_tiny_batch_throughput(self):
        for c in cases("full"):
            if not c.check_only and len(c.axes) == 1 and len(c.shape) > 1:
                self.assertGreaterEqual(min(c.shape[a] for a in c.axes), 128)


if __name__ == "__main__":
    unittest.main()
