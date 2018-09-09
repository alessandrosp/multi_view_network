"""Main module for multi_view_network, containing all Layers.

This module is based on End-to-End Multi-View Networks for Text Classification
by Hongyu Guo, Colin Cherry and Jiang Su (2017). Some of the details
in the paper were not explained with sufficient details, so I had to
make some guess work.

A few words about the terminology used in this module:
    - corpus: the set of all documents in a given experiment.
    - document: a piece of text, e.g. a tweet, an HTML page, a book.
    - token: often a word but not necessary, a small piece of text computed
        from a document.
    - embedding: a numerical representation for a piece of text.
    - embedded_(corpus|document|token): same as the terms defined above, but
        in embedding form. E.g. 'Hello world' could be a document made of the
        tokens 'hello' and 'world'. The embedded version of the document
        could be [[0, 1, 7], [9, 8, 23]].

Please, contact aless@ndro.xyz for any feedback.

Example:
    # Assuming embedded_corpus has already been converted
    # to np.array() of shape (num_documents, num_tokens, embeddings_dim).
    import multi_view_network

    model = multi_view_network.BuildMultiViewNetwork(
        embeddings_dim=3, hidden_units=16, dropout_rate=0, output_units=2)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    model.fit(embedded_corpus, labels, epochs=10, batch_size=2, verbose=0)
"""

import keras
import keras.backend as K
import keras.engine.topology
import tensorflow as tf

from .utils import check_shape


class SelectionLayer(keras.engine.topology.Layer):

    def __init__(self, **kwargs):
        super(SelectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings_dim = input_shape[2]
        self.major_w = self.add_weight(name='major_w',
                                       shape=(self.embeddings_dim,
                                              self.embeddings_dim),
                                       initializer='uniform',
                                       trainable=True)
        self.minor_w = self.add_weight(name='minor_w',
                                       shape=(1, self.embeddings_dim),
                                       initializer='uniform',
                                       trainable=True)
        super(SelectionLayer, self).build(input_shape)

    def _compute_m_coefficient(self, embedded_token):
        """Computes m-coefficient for a single embedding.

        Args:
            embedded_token: a tensor of shape (self.embeddings_dim,).

        Returns:
            A tensor of shape (1,).
        """
        partial = K.reshape(embedded_token, (self.embeddings_dim, 1))
        partial = K.dot(self.major_w, partial)
        partial = K.dot(self.minor_w, K.tanh(partial))
        # Before returning the final output we extract the first (and only)
        # element of the tensor to avoid returning a tensor of shape (1, 1).
        # This is because in tensor-based libraries there's a difference
        # between tensor of shapes (1) vs. (1, 1) though they are fundamentally
        # the same thing from a mathematical point of view.
        return partial[0]

    def _exp_if_not_zero(self, m_coefficient):
        """Computes exp() if m_coefficient is not 0, returns 0 otherwise."""
        return K.switch(
            K.equal(m_coefficient, 0),
            lambda: K.variable([0]),
            lambda: K.exp(m_coefficient))

    def _compute_d_coefficient(self, m_exp_coefficient):
        """Computes d-coefficient given exp(m_coefficient)."""
        return m_exp_coefficient / self.sum_m_exp_coefficients

    def _set_d_coefficients_as_diag(self, d_coefficients):
        """Sets d_coefficients as the diagonal of a 0-filled square matrix.

        Args:
            d_coefficients: tensor of shape (num_tokens, 1).

        Returns:
            A square matrix tensor of shape (num_tokens, num_tokens). All
            values in the tensor are 0s but for the diagonal.
        """
        placeholder = K.dot(d_coefficients, K.transpose(d_coefficients))
        zeros = K.zeros_like(placeholder)
        return tf.matrix_set_diag(zeros, K.reshape(d_coefficients, (-1, )))

    def _weighted_sum_of_embedded_tokens(
            self, d_coefficients_diag, embedded_document):
        """Computes the weighted sum of embedded tokens.

        Args:
            d_coefficients_diag: tensor of shape (num_tokens, num_tokens).
            embedded_document: tensor of shape (num_tokens, embeddings_dim).

        Returns:
            A tensor of shape (1, embeddings_dim).
        """
        weighted_sum = K.sum(
            K.dot(d_coefficients_diag, embedded_document), axis=0)
        return K.reshape(weighted_sum, (1, -1))

    def _compute_selection_output(self, embedded_document):
        """Computes the output of the Selection layer for a document."""
        num_tokens = K.int_shape(embedded_document)[0]

        m_coefficients = K.map_fn(
            self._compute_m_coefficient, embedded_document)
        check_shape(m_coefficients.get_shape().as_list(), [num_tokens, 1])

        m_exp_coefficients = K.map_fn(self._exp_if_not_zero, m_coefficients)
        self.sum_m_exp_coefficients = K.sum(m_exp_coefficients)
        check_shape(m_exp_coefficients.get_shape().as_list(), [num_tokens, 1])

        d_coefficients = K.map_fn(
            self._compute_d_coefficient, m_exp_coefficients)
        check_shape(d_coefficients.get_shape().as_list(), [num_tokens, 1])

        d_coefficients_diag = self._set_d_coefficients_as_diag(d_coefficients)
        check_shape(
            d_coefficients_diag.get_shape().as_list(),
            [num_tokens, num_tokens])

        weighted_sum = self._weighted_sum_of_embedded_tokens(
            d_coefficients_diag, embedded_document)
        check_shape(
            weighted_sum.get_shape().as_list(), [1, self.embeddings_dim])

        return weighted_sum

    def call(self, x):
        return K.map_fn(self._compute_selection_output, x)

    def compute_output_shape(self, input_shape):
        return (None, 1, self.embeddings_dim)


class ViewLayer(keras.engine.topology.Layer):
    """Implementation of the View structure in Guo et al. (2017).

    Args:
        view_index: int|str, either the index of the view as an integer
            or the keyword 'last'. This is because the first and the last
            views in the network behaves very differently from the other
            ones.
    """

    def __init__(self, view_index, **kwargs):
        self.view_index = view_index
        super(ViewLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if (self.view_index == 1 or self.view_index == 'Last'):
            if not isinstance(input_shape, tuple):
                raise TypeError(
                    'First and last view should have only 1 input.')
            trainable = False
            self.embeddings_dim = input_shape[2]
            self.size_stacked_selections = 1
        else:
            if not isinstance(input_shape, list):
                raise TypeError(
                    'Views other than first and last should have'
                    'multiple inputs')
            trainable = True
            self.embeddings_dim = input_shape[0][2]
            self.size_stacked_selections = self.view_index*self.embeddings_dim

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.embeddings_dim,
                                             self.size_stacked_selections),
                                      initializer='uniform',
                                      trainable=trainable)
        super(ViewLayer, self).build(input_shape)

    def _stack_selections(self, x):
        """Stacks the various vectors vertically.

        Here x is a list of tensors. Each tensor has
        shape (batch_size, 1, embeddings_dim). First, the various
        selections are concatenated. The output of the concatenation
        is then a tensor (not a list of tensors as per x) with
        shape (batch_size, 1, view_index*embeddings_dim). Lastly,
        this tensor is reshaped to be vertical (and the 1 dimension is lost).

        Args:
            x: [tensor], list of selections.

        Returns:
            A tensor of shape (view_index*embeddings_dim, batch_size).
        """
        concatenated_selections = K.concatenate(x)
        return K.reshape(
            concatenated_selections, (self.size_stacked_selections, -1))

    def call(self, x):
        # The first and the last view in the network simply pass the
        # output of the first or last selection forward.
        if self.view_index == 1 or self.view_index == 'Last':
            return K.reshape(x, (-1, self.embeddings_dim))

        self.batch_size = K.int_shape(x[0])[0]
        stacked_selections = self._stack_selections(x)
        check_shape(
            stacked_selections.get_shape().as_list(),
            [self.size_stacked_selections, self.batch_size])
        output = K.reshape(
            K.tanh(K.dot(self.kernel, stacked_selections)),
            (-1, self.embeddings_dim))

        return output

    def compute_output_shape(self, input_shape):
        return (None, self.embeddings_dim)


def BuildMultiViewNetwork(
        embeddings_dim, hidden_units, dropout_rate, output_units):
    """Builds a Multi-View Network as specified in Guo et al. (2017).

    Args:
        embeddings_dim: int, the dimensionality of the embedding space.
        hidden_units: int, the number of units in the "hidden" layer, i.e.
            the last layer before the softmax output layer.
        dropout_rate: float, probability for a unit to be dropped.
        output_units: int, the number of units in the output layer. Given
            than the output layer uses softmax as an activation function,
            this number should match the number of classes of the
            classification task.

    """
    inputs = keras.layers.Input(shape=(None, embeddings_dim))
    s1 = SelectionLayer(name='s1')(inputs)
    s2 = SelectionLayer(name='s2')(inputs)
    s3 = SelectionLayer(name='s3')(inputs)
    s4 = SelectionLayer(name='s4')(inputs)
    v1 = ViewLayer(view_index=1, name='v1')(s1)
    v2 = ViewLayer(view_index=2, name='v2')([s1, s2])
    v3 = ViewLayer(view_index=3, name='v3')([s1, s2, s3])
    v4 = ViewLayer(view_index='Last', name='v4')(s4)
    concatenation = keras.layers.concatenate(
        [v1, v2, v3, v4], name='concatenation')
    fully_connected = keras.layers.Dense(
        units=hidden_units, name='fully_connected')(concatenation)
    dropout = keras.layers.Dropout(rate=dropout_rate)(fully_connected)
    softmax = keras.layers.Dense(
        units=output_units, activation='softmax',
        name='softmax')(dropout)

    return keras.models.Model(inputs=inputs, outputs=softmax)
