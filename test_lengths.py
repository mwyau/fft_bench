import unittest
from lengths import length_class, transform_lengths

class TestLengths(unittest.TestCase):
    def test_factorization(self):
        self.assertEqual(length_class((4096,)), "power of 2")
        self.assertEqual(length_class((257,)), "prime")
        self.assertEqual(length_class((4095,)), "odd mixed radix (largest prime >7)")
        self.assertEqual(length_class((243,)), "power of 3")

    def test_axes(self):
        self.assertEqual(transform_lengths(dict(shape=[8, 128, 257], axes=[-1, 1], lengths=[])), (257, 128))
