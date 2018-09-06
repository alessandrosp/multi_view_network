import unittest

import keras.backend as K
import numpy as np

import multi_view_network


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

    def test_compute_m_exp_coefficients_returns_correct_shape(self):
        selection_layer = self._get_selection_layer()
        x = K.variable([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        m_exp_coefficients = selection_layer._compute_m_exp_coefficients(x)
        m_exp_coefficients_shape = m_exp_coefficients.get_shape().as_list()
        self.assertEqual(m_exp_coefficients_shape, [1, 4])

    def test_compute_d_coefficient_returns_correct_value(self):
        selection_layer = self._get_selection_layer()
        d_coefficient = selection_layer._compute_d_coefficient(10)
        self.assertEqual(d_coefficient, 5)

    def test_compute_d_coefficients_returns_correct_shape(self):
        selection_layer = self._get_selection_layer()
        m_exp_coefficients = K.variable([[1, 1, 1, 1]])
        d_coefficients = selection_layer._compute_d_coefficients(
            m_exp_coefficients)
        d_coefficients_shape = d_coefficients.get_shape().as_list()
        self.assertEqual(d_coefficients_shape, [1, 4])

    def test_set_d_coefficients_as_diag_returns_correct_matrix(self):
        selection_layer = self._get_selection_layer()
        d_coefficients = K.variable([[1, 2, 3]])
        d_coefficients_diag = K.eval(
            selection_layer._set_d_coefficients_as_diag(d_coefficients))
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        self.assertTrue(np.array_equal(d_coefficients_diag, expected))

    def test_set_d_coefficients_as_diag_returns_correct_shape(self):
        selection_layer = self._get_selection_layer()
        d_coefficients = K.variable([[1, 2, 3, 4, 5]])
        d_coefficients_diag = selection_layer._set_d_coefficients_as_diag(
            d_coefficients)
        d_coefficients_diag_shape = d_coefficients_diag.get_shape().as_list()
        self.assertEqual(d_coefficients_diag_shape, [5, 5])

    def test_compute_layer_output_returns_correct_values(self):
        selection_layer = self._get_selection_layer()
        # Just as a remainder: a square matrix with all 1s in the diagonal
        # and 0s in all other cells is an identity matrix, the equivalent
        # of the number 1 in the multiplication. Because of this
        # the output of _compute_layer_output() is going to be just
        # the vertical sum of values in x.
        d_coefficients_diag = K.variable([[1, 0], [0, 1]])
        x = K.variable([[1, 2, 3], [5, 6, 7]])
        layer_output = K.eval(
            selection_layer._compute_layer_output(d_coefficients_diag, x))
        expected = np.array([6, 8, 10])  # Vertical sum of x.
        self.assertTrue(np.array_equal(layer_output, expected))

    def test_compute_layer_output_returns_correct_shape(self):
        selection_layer = self._get_selection_layer()
        # d_coefficients_diag is a square matrix with shape equals
        # to the number of tokens in x. On the other hand, x has
        # shape equals to (num_tokens, embeddings_dim). In this
        # suite embeddings_dim is set to a fixed value
        # in _get_selection_layer().
        d_coefficients_diag = K.variable([[1, 0], [0, 1]])
        x = K.variable([[1, 2, 3], [5, 6, 7]])
        layer_output = selection_layer._compute_layer_output(
            d_coefficients_diag, x)
        layer_output_shape = layer_output.get_shape().as_list()
        self.assertEqual(layer_output_shape, [3])


class TestViewLayer(unittest.TestCase):

    def _get_view_layer(self):
        view_layer = multi_view_network.ViewLayer(view_index=2)
        # Kernel has to have shape (embeddings_dim, view_index*embeddings_dim).
        view_layer.kernel = K.variable(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        view_layer.embeddings_dim = 3
        view_layer.size_stacked_selections = 6
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


if __name__ == '__main__':
    unittest.main()
