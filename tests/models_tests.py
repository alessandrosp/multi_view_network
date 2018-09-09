import os
import unittest

import keras.backend as K
import numpy as np

import multi_view_network

# Suppress the Your CPU supports instructions that warning.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TestSelectionLayer(unittest.TestCase):

    def _get_selection_layer(self):
        selection_layer = multi_view_network.SelectionLayer()
        # Implenets a simple (4 x 3) matrix for the weights.
        selection_layer.major_w = K.variable([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        selection_layer.minor_w = K.variable([[1, 1, 1]])
        selection_layer.embeddings_dim = 3
        selection_layer.sum_m_exp_coefficients = 2
        return selection_layer

    def test_major_w_selects_first_three_rows(self):
        selection_layer = self._get_selection_layer()
        major_w = K.eval(selection_layer.major_w)
        expected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        self.assertTrue(np.array_equal(major_w, expected))

    def test_minor_w_selects_last_row(self):
        selection_layer = self._get_selection_layer()
        minor_w = K.eval(selection_layer.minor_w)
        expected = np.array([[1, 1, 1]])
        self.assertTrue(np.array_equal(minor_w, expected))

    def test_compute_m_coefficient_outputs_correct_value(self):
        selection_layer = self._get_selection_layer()
        embedding = K.variable([1, 1, 1])
        m_coefficient = K.eval(
            selection_layer._compute_m_coefficient(embedding))
        expected = np.array([2.9851642])
        self.assertTrue(np.isclose(m_coefficient, expected))

    def test_compute_m_coefficient_when_embedding_is_zeros(self):
        selection_layer = self._get_selection_layer()
        embedding = K.variable([0, 0, 0])
        m_coefficient = K.eval(
            selection_layer._compute_m_coefficient(embedding))
        expected = np.array([0])
        self.assertTrue(np.isclose(m_coefficient, expected))

    def test_compute_m_coefficient_works_with_map_fn(self):
        x = K.variable([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        selection_layer = self._get_selection_layer()
        m_coefficients = K.eval(
            K.map_fn(selection_layer._compute_m_coefficient, x))
        expected = np.array([2.9851642, 2.9851642, 2.9851642, 2.9851642])
        are_values_close = np.isclose(m_coefficients, expected)
        self.assertTrue(np.all(are_values_close))

    def test_exp_if_not_zero_returns_exp_when_not_zero(self):
        selection_layer = self._get_selection_layer()
        m_coefficient = K.variable([1])
        output = K.eval(selection_layer._exp_if_not_zero(m_coefficient))[0]
        self.assertTrue(np.isclose(output, 2.7182817))

    def test_exp_if_not_zero_returns_zero_when_zero(self):
        selection_layer = self._get_selection_layer()
        m_coefficient = K.variable([0])
        output = K.eval(selection_layer._exp_if_not_zero(m_coefficient))
        self.assertEqual(output, 0)

    def test_compute_d_coefficient_returns_correct_value(self):
        selection_layer = self._get_selection_layer()
        d_coefficient = selection_layer._compute_d_coefficient(10)
        self.assertEqual(d_coefficient, 5)

    def test_set_d_coefficients_as_diag_returns_correct_matrix(self):
        selection_layer = self._get_selection_layer()
        d_coefficients = K.variable([[1], [2], [3]])
        d_coefficients_diag = K.eval(
            selection_layer._set_d_coefficients_as_diag(d_coefficients))
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(np.array_equal(d_coefficients_diag, expected))

    def test_set_d_coefficients_as_diag_returns_correct_shape(self):
        selection_layer = self._get_selection_layer()
        d_coefficients = K.variable([[1], [2], [3], [4], [5]])
        d_coefficients_diag = selection_layer._set_d_coefficients_as_diag(
            d_coefficients)
        d_coefficients_diag_shape = d_coefficients_diag.get_shape().as_list()
        self.assertEqual(d_coefficients_diag_shape, [5, 5])

    def test_weighted_sum_of_embedded_tokens_returns_correct_values(self):
        selection_layer = self._get_selection_layer()
        # Just as a remainder: a square matrix with all 1s in the diagonal
        # and 0s in all other cells is an identity matrix, the equivalent
        # of the number 1 in the multiplication. Because of this
        # the output of _compute_layer_output() is going to be just
        # the vertical sum of values in x.
        d_coefficients_diag = K.variable([[1, 0], [0, 1]])
        x = K.variable([[1, 2, 3], [5, 6, 7]])
        weighted_sum = K.eval(
            selection_layer._weighted_sum_of_embedded_tokens(
                d_coefficients_diag, x))
        expected = np.array([[6, 8, 10]])  # Vertical sum of x.
        self.assertTrue(np.array_equal(weighted_sum, expected))

    def test_weighted_sum_of_embedded_tokens_returns_correct_shape(self):
        selection_layer = self._get_selection_layer()
        # d_coefficients_diag is a square matrix with shape equals
        # to the number of tokens in x. On the other hand, x has
        # shape equals to (num_tokens, embeddings_dim). In this
        # suite embeddings_dim is set to a fixed value
        # in _get_selection_layer().
        d_coefficients_diag = K.variable([[1, 0], [0, 1]])
        x = K.variable([[1, 2, 3], [5, 6, 7]])
        weighted_sum = selection_layer._weighted_sum_of_embedded_tokens(
            d_coefficients_diag, x)
        weighted_sum_shape = weighted_sum.get_shape().as_list()
        self.assertEqual(weighted_sum_shape, [1, 3])

    def test_compute_selection_output_same_output_when_padded(self):
        selection_layer = self._get_selection_layer()
        embedded_document = K.variable([[1, 2, 3], [5, 6, 7]])
        embedded_document_padded = K.variable(
            [[1, 2, 3], [5, 6, 7], [0, 0, 0], [0, 0, 0]])
        weighted_sum = K.eval(
            selection_layer._compute_selection_output(embedded_document))
        weighted_sum_padded = K.eval(
            selection_layer._compute_selection_output(
                embedded_document_padded))
        self.assertTrue(np.array_equal(weighted_sum, weighted_sum_padded))


class TestViewLayer(unittest.TestCase):

    def _get_view_layer(self):
        view_layer = multi_view_network.ViewLayer(view_index=2)
        # Kernel has to have shape (embeddings_dim, view_index*embeddings_dim).
        view_layer.kernel = K.variable(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        view_layer.embeddings_dim = 3
        view_layer.size_stacked_selections = 6
        view_layer.batch_size = 1
        return view_layer

    def test_stack_selections_returns_right_shape(self):
        view_layer = self._get_view_layer()
        x = [K.variable([[1, 2, 3]]), K.variable([[4, 5, 6]])]
        stacked_selections = view_layer._stack_selections(x)
        stacked_selections_shape = stacked_selections.get_shape().as_list()
        expected = [6, 1]
        self.assertEqual(stacked_selections_shape, expected)

    def test_stack_selections_return_correct_values(self):
        view_layer = self._get_view_layer()
        x = [K.variable([[1, 2, 3]]), K.variable([[4, 5, 6]])]
        stacked_selections = K.eval(view_layer._stack_selections(x))
        expected = np.array([[1], [2], [3], [4], [5], [6]])
        self.assertTrue(np.array_equal(stacked_selections, expected))


class TestMiltiViewNetwork(unittest.TestCase):

    def test_multi_view_network_trains_fine(self):
        labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        embedded_corpus = np.array([
            [
                [1, 1, 1], [2, 2, 2]
            ],
            [
                [1, 1, 1], [4, 4, 4]
            ],
            [
                [1, 1, 1], [6, 6, 6]
            ],
            [
                [1, 1, 1], [8, 8, 8]
            ],
        ])
        model = multi_view_network.BuildMultiViewNetwork(
            embeddings_dim=3, hidden_units=16, dropout_rate=0, output_units=2)
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        model.fit(embedded_corpus, labels, epochs=10, batch_size=2, verbose=0)
        self.assertTrue(model.build)

    def test_multi_view_network_trains_fine_with_padded_input(self):
        labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        embedded_corpus = [
            [
                [1, 1, 1], [2, 2, 2]
            ],
            [
                [1, 1, 1], [2, 2, 2], [4, 4, 4], [6, 6, 6]
            ],
            [
                [1, 1, 1], [2, 2, 2], [4, 4, 4], [6, 6, 6], [8, 8, 8]
            ],
            [
                [1, 1, 1], [2, 2, 2]
            ],
        ]
        padded_corpus = np.array(
            multi_view_network.pad_embedded_corpus(embedded_corpus, 3))
        model = multi_view_network.BuildMultiViewNetwork(
            embeddings_dim=3, hidden_units=16, dropout_rate=0, output_units=2)
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        model.fit(padded_corpus, labels, epochs=10, batch_size=2, verbose=0)
        self.assertTrue(model.build)


if __name__ == '__main__':
    unittest.main()
