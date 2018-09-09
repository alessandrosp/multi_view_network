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

    def test_pad_embedded_corpus_returns_right_sizes(self):
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
        sizes = [len(lst) for lst in padded_corpus]
        expected = [3, 3, 3]
        self.assertEqual(sizes, expected)

    def test_pad_embedded_corpus_does_not_change_input(self):
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
        _ = multi_view_network.pad_embedded_corpus(
            embedded_corpus, 2)
        sizes = [len(lst) for lst in embedded_corpus]
        expected = [1, 2, 3]
        self.assertEqual(sizes, expected)

    def test_cap_embedded_corpus_returns_right_sizes(self):
        embedded_corpus = [
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
        capped_corpus = multi_view_network.cap_embedded_corpus(
            embedded_corpus, cap_size=2)
        sizes = [len(lst) for lst in capped_corpus]
        expected = [2, 2]
        self.assertEqual(sizes, expected)

    def test_cap_embedded_corpus_inferres_correct_cap_size(self):
        embedded_corpus = [
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
        capped_corpus = multi_view_network.cap_embedded_corpus(embedded_corpus)
        sizes = [len(lst) for lst in capped_corpus]
        expected = [2, 2]
        self.assertEqual(sizes, expected)

    def test_cap_embedded_corpus_excludes_if_below_cap(self):
        embedded_corpus = [
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
        capped_corpus = multi_view_network.cap_embedded_corpus(
            embedded_corpus, cap_size=3, exclude_if_below_cap=True)
        sizes = [len(lst) for lst in capped_corpus]
        expected = [3]
        self.assertEqual(sizes, expected)

    def test_cap_embedded_corpus_does_not_change_input(self):
        embedded_corpus = [
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
        _ = multi_view_network.cap_embedded_corpus(
            embedded_corpus, cap_size=3, exclude_if_below_cap=True)
        sizes = [len(lst) for lst in embedded_corpus]
        expected = [2, 3]
        self.assertEqual(sizes, expected)
