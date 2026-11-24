import unittest

from resolve_sources import UPSTREAM, ensure_reference


class TestReference(unittest.TestCase):
    def test_reuses_upstream(self):
        config = {"ducc": [{"name": "ducc", "repo": UPSTREAM, "ref": "ducc0"}]}
        ensure_reference(config)
        self.assertEqual(len(config["ducc"]), 1)
        self.assertTrue(config["ducc"][0]["reference"])

    def test_preserves_custom_source(self):
        config = {"ducc": [{"name": "ducc", "repo": UPSTREAM, "ref": "old"}]}
        ensure_reference(config)
        self.assertEqual(len(config["ducc"]), 2)
        self.assertEqual(config["ducc"][0]["name"], "ducc-upstream")
        self.assertEqual(config["ducc"][0]["levels"], [1])
        self.assertEqual(config["ducc"][1]["ref"], "old")
