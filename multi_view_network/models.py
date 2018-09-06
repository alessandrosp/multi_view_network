import keras
import keras.backend as K
import keras.engine.topology
import tensorflow as tf


class SelectionLayer(keras.engine.topology.Layer):
    """Selection Layer for the Multi-View Network."""

    def __init__(self, **kwargs):
        super(SelectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initialize the weights for the layer.

        The shape of the weights matrix is computed from the size of the
        embeddings. Specifically, the matrix can be though as composed by
        two distinct elements vertically stacked:
            - A square matrix of size (embeddings_dim, embeddings_dim).
            - A vector of size (embeddings_dim, 1).

        When self.call() is execute, the two elements are used as part of the
        same equation.

        Args:
            input_shape: (num_rows, num_cols), the size of the bag-of-words
                feature matrix (as defined in the original paper). Note
                that num_cols is the same as the size of the embeddings.
        """
        self.embeddings_dim = input_shape[1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.embeddings_dim+1,
                                             self.embeddings_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(SelectionLayer, self).build(input_shape)

    def _get_major_w(self):
        """Gets the first self.embeddings_dim rows of the weight matrix.

        In equation (3) two different set of weights are used to compute
        the m-coefficient for a given embedding. One of the two weights
        is referred to as w(i, h), while the other one as W(i, h). The
        first set of weights is an vector, while the second one is a
        square matrix of the same size as the dimensionality of the
        embeddings. Here, we refer to the vector-weights as minor_w and
        the matrix-weights as major w.

        Returns:
            A square matrix (tensor) of size self.embeddings_dim.
        """
        return K.slice(
            self.kernel, (0, 0), (self.embeddings_dim, self.embeddings_dim))

    def _get_minor_w(self):
        """Gets the last row of the weight matrix.

        In equation (3) two different set of weights are used to compute
        the m-coefficient for a given embedding. One of the two weights
        is referred to as w(i, h), while the other one as W(i, h). The
        first set of weights is an vector, while the second one is a
        square matrix of the same size as the dimensionality of the
        embeddings. Here, we refer to the vector-weights as minor_w and
        the matrix-weights as major w.

        Returns:
            A vector (tensor) of size (1, self.embeddings_dim).
        """
        return K.slice(
            self.kernel, (self.embeddings_dim, 0), (1, self.embeddings_dim))

    def _compute_m_coefficient(self, embedding):
        """Computes the m-coefficient for an embedding.

        This method computes the output of equation (3). In the paper, this
        coefficient is represented with the letter m, thus the
        name m-coefficient.

        Args:
            embedding: a tensor of size (1, self.embeddings_dim).

        Returns:
            The m-coefficient for the embedding as tensor of one element.
        """
        partial = K.reshape(embedding, (1, self.embeddings_dim))
        partial = K.dot(partial, self._get_major_w())
        partial = K.reshape(K.tanh(partial), (self.embeddings_dim, 1))
        partial = K.dot(self._get_minor_w(), partial)
        # Before returning the final output we extract the first (and only)
        # element of the tensor to avoid returning a tensor of shape (1, 1).
        # This is because in tensor-based libraries there's a difference
        # between tensor of shapes (1) vs. (1, 1) though they are fundamentally
        # the same thing from a mathematical point of view.
        return partial[0]

    def _compute_m_exp_coefficients(self, x):
        """Computes the exp of the m-coeeficients for all tokens in x.

        The details of this passage can be found in (2). The function also
        sets self.sum_m_exp_coefficients so that it can be used to compute
        d-coefficients later on (equation 2).

        Args:
            x: tensor of shape (num_tokens, embeddings_dim). Each row in the
                matrix is an embedding for a token.

        Returns:
            A tensor of shape (1, num_tokens) containing the m-coefficients
            for each of the token.
        """
        m_coefficients = K.map_fn(self._compute_m_coefficient, x)
        m_exp_coefficients = K.map_fn(K.exp, m_coefficients)
        # Before the layer is actually compiled and trained x is going
        # to have a None number of tokens.
        num_tokens = x.get_shape().as_list()[0] or 1
        self.sum_m_exp_coefficients = K.sum(m_exp_coefficients)
        return K.reshape(m_exp_coefficients, (1, num_tokens))

    def _compute_d_coefficient(self, m_exp_coefficient):
        """Computes the d-coefficient for a single token.

        Given the exp of the m-coefficient for a give token this method
        computes the d-coefficient. Details are given in (2).

        Args:
            m_exp_coefficient: a scalar-tensor.

        Returns:
            A scalar-tensor.
        """
        return m_exp_coefficient / self.sum_m_exp_coefficients

    def _compute_d_coefficients(self, m_exp_coefficients):
        """Computes the d-coefficients for the tokens.

        Given a tensor of the exp of m-coefficients for all tokens in the
        document, it computes all relevant d-coefficients (as per 2).

        Args:
            m_exp_coefficients: a tensor of shape (1, num_tokens).

        Returns:
            A tensor of shape (1, num_tokens).
        """
        d_coefficients = K.map_fn(
            self._compute_d_coefficient, m_exp_coefficients)
        num_tokens = m_exp_coefficients.get_shape().as_list()[1]
        return K.reshape(d_coefficients, (1, num_tokens))

    def _set_d_coefficients_as_diag(self, d_coefficients):
        """Set the scores in d_coefficients as the diag of a 0-filled matrix.

        First, a square matrix of zeros is created with the same size as
        d_coefficients. Then d_coefficients is set to be the diagonal of
        the square matrix. This is done in order to make the computation
        for (1) more efficient (this way it can be done as a simple dot
        product rather than having to iterate through all the embeddings).

        Args:
            d_coefficients: a tensor of shape (1, num_tokens).

        Returns:
            A square matrix of size (num_tokens, num_tokens). All the values
            are 0s but for the diagonal which is the same as d_coefficients.
        """
        num_tokens = d_coefficients.get_shape().as_list()[1]
        zeros = K.zeros(shape=(num_tokens, num_tokens))
        return tf.matrix_set_diag(
            zeros, K.reshape(d_coefficients, (num_tokens,)))

    def _compute_layer_output(self, d_coefficients_diag, x):
        """Computes the final output of the Selection layer.

        More details available in (1).

        Args:
            d_coefficients: a tensor of shape (1, num_tokens).
            x: tensor of shape (num_tokens, embeddings_dim). Each row in the
                matrix is an embedding for a token.

        Returns:
            A tensor of shape (1, embeddings_dim).
        """
        sum = K.sum(K.dot(d_coefficients_diag, x), axis=0)
        return K.reshape(sum, (1, self.embeddings_dim))

    def call(self, x):
        """Computes the output of the Selection and pass it to the next layer.

        Before the output of the selection can be passed to the next layer
        a series of coefficients need to be computed. These coefficients
        are simply referred to as d and m in Guo et al. (2017). I decided
        to keep the same names for consistency. Also note that these
        coefficients are all computed at the word level (meaning there
        a d and m coefficient for each embedding in x).

        Fundamentally, the weights defined in self.build() determine
        how much a word is going to influece the final vector outputted (not
        at all dissimilar to the way a set of coeeficients determine the
        final output in a weighted average).

        Args:
            x: a tensor, where the each row represents a word and the
                number of columns is determined by the size of the
                embedding space.

        Returns:
            A tensor of shape (1, embeddings_dim).
        """
        m_exp_coefficients = self._compute_m_exp_coefficients(x)
        d_coefficients = self._compute_d_coefficients(m_exp_coefficients)
        d_coefficients_diag = self._set_d_coefficients_as_diag(d_coefficients)
        return self._compute_layer_output(d_coefficients_diag, x)

    def compute_output_shape(self, input_shape):
        return (1, self.embeddings_dim)


class ViewLayer(keras.engine.topology.Layer):
    """View Layer for the Multi-View Network.

    Args:
        view_index: int|str, either the index of view (starting
            from 1) or the keyword 'last'. This is because
            the last view (as the first one) behaves very
            differently from the other views.
    """

    def __init__(self, view_index, **kwargs):
        self.view_index = view_index
        super(ViewLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initialize the weights for the layer.

        A few things to keep in mind:
            - If view_index is 1 or 'Last', then the weights
              are not trainable (in fact, they're useless as the layer
              then only output the input).
            - If view_index is 1 or 'Last' input is expected to be a tensor
              and input_shape a tuple (i.e. nrows x ncols). In the other cases
              the input should be a list of tensors and input_shape a list
              of tuples.

        Args:
            input_shape: ()|[()], where each tuple is (1, embeddings_dim).
        """
        if (self.view_index == 1 or self.view_index == 'Last'):
            if not isinstance(input_shape, tuple):
                raise TypeError(
                    'First and last view should have only 1 input.')
            trainable = False
            self.embeddings_dim = input_shape[1]
            self.size_stacked_selections = 1
        else:
            if not isinstance(input_shape, list):
                raise TypeError(
                    'Views other than first and last should have'
                    'multiple inputs')
            trainable = True
            self.embeddings_dim = input_shape[0][1]
            self.size_stacked_selections = self.view_index*self.embeddings_dim

        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.embeddings_dim,
                                             self.size_stacked_selections),
                                      initializer='uniform',
                                      trainable=trainable)
        super(ViewLayer, self).build(input_shape)

    def _stack_selections(self, x):
        """Vertically stacks selections into one tensor.

        First, concatenates all the selections in x horizontally. This means
        concatenated_selections will have shapve (1 x len(x)*embeddings_dim).
        Then, the tensor is reshaped so for it to be vertical.

        Args:
            x: a list of tensors. Each tensor has shape (1, embeddings_dim).

        Returns:
            A tensor of shape (len(x)*embeddings_dim, 1).
        """
        concatenated_selections = K.concatenate(x)

        return K.reshape(
            concatenated_selections, (self.size_stacked_selections, 1))

    def call(self, x):
        """Computes the output of the View and pass it to the next layer.

        Args:
            x: a tensor or a list of tensors. Each tensor has
                shape (1, embeddings_dim).

        Returns:
            A tensor of shape (1, embeddings_dim).
        """
        # The first and the last view in the network simply pass the
        # output of the first or last selection forward.
        if self.view_index == 1 or self.view_index == 'Last':
            return x

        stacked_selections = self._stack_selections(x)

        # At this stage the output is still vertical (same shape as
        # the individual selections, e.g. embeddings_dim x 1).
        output = K.dot(self.kernel, stacked_selections)

        return K.reshape(output, (1, self.embeddings_dim))

    def compute_output_shape(self, input_shape):
        return (1, self.embeddings_dim)


def BuildStandardMultiViewNetwork(embeddings_dim, hidden_units, output_units):
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
    inputs = keras.layers.Input(shape=(embeddings_dim,))
    s1 = SelectionLayer(name='s1')(inputs)
    s2 = SelectionLayer(name='s2')(inputs)
    s3 = SelectionLayer(name='s3')(inputs)
    s4 = SelectionLayer(name='s4')(inputs)
    v1 = ViewLayer(view_index=1, name='v1')(s1)
    v2 = ViewLayer(view_index=2, name='v2')([v1, s2])
    v3 = ViewLayer(view_index=3, name='v3')([v1, v2, s3])
    v4 = ViewLayer(view_index='Last', name='v4')(s4)
    concatenation = keras.layers.concatenate([v1, v2, v3, v4])
    fully_connected = keras.layers.Dense(units=hidden_units)(concatenation)
    softmax = keras.layers.Dense(
        units=output_units, activation='softmax')(fully_connected)

    return keras.models.Model(inputs=inputs, outputs=softmax)


model = BuildStandardMultiViewNetwork(
    embeddings_dim=3, hidden_units=16, output_units=1)
