import unittest

import multi_view_network


class TestUtils(unittest.TestCase):

    def test_check_shape_works_fine_when_same_shape(self):
        to_check = [1, 1]
        expected = [1, 1]
        multi_view_network.check_shape(to_check, expected)
        # If above didn't raise any error, we assert True is True.
        self.assertTrue(True)

    def test_check_shape_raises_error_when_unexpected_shape(self):
        to_check = [1, 1]
        expected = [1, 2]
        with self.assertRaises(AssertionError):
            multi_view_network.check_shape(to_check, expected)

    def test_pad_embedded_corpus_returns_right_structure(self):
        embedded_corpus = [
            [
                [0, 0]
            ],
            [
                [0, 0],
                [1, 1]
            ],
            [
                [0, 0],
                [1, 1],
                [2, 1]
            ]
        ]
        padded_corpus = multi_view_network.pad_embedded_corpus(
            embedded_corpus, 2)
        lengths = [len(lst) for lst in embedded_corpus]
        expected = [3, 3, 3]
        self.assertEquals(lengths, expected)
