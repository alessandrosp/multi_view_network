import keras
import keras.backend as K
import keras.engine.topology
import tensorflow as tf


def assert_shape(to_check, expected):
    try:
        assert to_check == expected
    except AssertionError:
        message = ('Shape doesn\'t meet expecations. I\'m '
                   'afraid {to_check} != {expected}'.format(to_check=to_check,
                                                            expected=expected))
        raise AssertionError(message)


def pad_embedded_corpus(embedded_corpus, embeddings_dim):
    max_num_tokens = 1
    for embedded_document in embedded_corpus:
        if len(embedded_document) > max_num_tokens:
            max_num_tokens = len(embedded_document)

    padded_corpus = []
    for embedded_document in embedded_corpus:
        for _ in range(max_num_tokens-len(embedded_document)):
            embedded_document.append([0]*embeddings_dim)
        padded_corpus.append(embedded_document)

    return padded_corpus


class SelectionLayer(keras.engine.topology.Layer):
    """Selection Layer for the Multi-View Network."""

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
        return K.switch(
            K.equal(m_coefficient, 0),
            lambda: K.variable([0]),
            lambda: K.exp(m_coefficient))

    def _compute_d_coefficient(self, m_exp_coefficient):
        return m_exp_coefficient / self.sum_m_exp_coefficients

    def _set_d_coefficients_as_diag(self, d_coefficients):
        placeholder = K.dot(d_coefficients, K.transpose(d_coefficients))
        zeros = K.zeros_like(placeholder)
        return tf.matrix_set_diag(zeros, K.reshape(d_coefficients, (-1, )))

    def _weighted_sum_of_embedded_tokens(self, d_coefficients_diag, embedded_document):
        weighted_sum = K.sum(
            K.dot(d_coefficients_diag, embedded_document), axis=0)
        return K.reshape(weighted_sum, (1, -1))

    def _compute_selection_output(self, embedded_document):

        num_tokens = K.int_shape(embedded_document)[0]

        # Compute m_coefficients
        m_coefficients = K.map_fn(
            self._compute_m_coefficient, embedded_document)
        assert_shape(m_coefficients.get_shape().as_list(), [num_tokens, 1])

        # Compute m_exp_coefficients
        # m_exp_coefficients = K.map_fn(K.exp, m_coefficients)
        m_exp_coefficients = K.map_fn(self._exp_if_not_zero, m_coefficients)
        self.sum_m_exp_coefficients = K.sum(m_exp_coefficients)
        assert_shape(m_exp_coefficients.get_shape().as_list(), [num_tokens, 1])

        # Comoute d_coefficients
        d_coefficients = K.map_fn(
            self._compute_d_coefficient, m_exp_coefficients)
        assert_shape(d_coefficients.get_shape().as_list(), [num_tokens, 1])

        # Compite
        d_coefficients_diag = self._set_d_coefficients_as_diag(d_coefficients)
        assert_shape(
            d_coefficients_diag.get_shape().as_list(),
            [num_tokens, num_tokens])

        weighted_sum = self._weighted_sum_of_embedded_tokens(
            d_coefficients_diag, embedded_document)
        assert_shape(
            weighted_sum.get_shape().as_list(), [1, self.embeddings_dim])

        return weighted_sum

    def call(self, x):
        return K.map_fn(self._compute_selection_output, x)

    def compute_output_shape(self, input_shape):
        return (None, 1, self.embeddings_dim)


class ViewLayer(keras.engine.topology.Layer):

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
        assert_shape(
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


model = BuildMultiViewNetwork(
    embeddings_dim=3, hidden_units=16, dropout_rate=0, output_units=2)


if __name__ == '__main__':
    import numpy as np
    EMBEDDINGS_MODEL = {
        'alice': [1, 1, 1],
        'bob': [1, 1, 1],
        'charlie': [1, 1, 1],
        'dany': [1, 1, 1],
        'loves': [2, 2, 2],
        'hates': [4, 4, 4],
        'koalas': [8, 8, 8]
    }
    corpus = [
        'Alice loves',
        'Bob loves',
        'Charlie hates',
        'Dany hates koalas'
    ]
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    tokenized_corpus = [
        keras.preprocessing.text.text_to_word_sequence(document)
        for document in corpus]
    embedded_corpus = []
    for tokenized_document in tokenized_corpus:
        embedded_document = []
        for token in tokenized_document:
            embedded_document.append(EMBEDDINGS_MODEL[token])
        embedded_corpus.append(embedded_document)

    data = np.array(pad_embedded_corpus(embedded_corpus, 3))
    model.compile(
        optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(data, labels, epochs=100, batch_size=4)
