# Multi-View Network in Keras

This package is based on [End-to-End Multi-View Networks for Text Classification](https://arxiv.org/abs/1704.05907) by Hongyu Guo, Colin Cherry and Jiang Su (2017). The overall architecture of the Multi-View Network (MVN) was not really explained in painstaking details in the paper, so I had to make some guess work.

Feel free reach out to me at aless@ndro.xyz with any feedback.

# Basic Usage

Assuming you have your corpus prepared as a list of documents, each represented by a list of embeddings (one per token), you can train the MVN this way:

```python
import multi_view_network
import numpy as np

# Very important: the documents in embedded_corpus **need** to have
# the same number of embedded_tokens. If this is not the case
# you can use multi_view_network.pad_embedded_corpus() to pad
# the documents with 0-filled mock embeddings.
data = np.array(embedded_corpus)

# The output of the MVN is softmaxed so it's important to
# make sure the labels are one-hot encoded.
labels = np.array([[0, 1], [0, 1], [1, 0], etc.])

model = multi_view_network.BuildMultiViewNetwork(
    embeddings_dim=300, hidden_units=16, dropout_rate=0.2, output_units=2)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(data, labels, epochs=200, batch_size=32)
```

# More Complex Architectures

The `models.py` module contains all the necessary Layers to build MVNs of arbitrary size and complexity. For example:

```python
import multi_view_network

embeddings_dim = 300
hidden_units = 64
output_units = 2

inputs = keras.layers.Input(shape=(None, embeddings_dim))
s1 = SelectionLayer(name='s1')(inputs)
s2 = SelectionLayer(name='s2')(inputs)
s3 = SelectionLayer(name='s3')(inputs)
s4 = SelectionLayer(name='s4')(inputs)
s5 = SelectionLayer(name='s5')(inputs)
s6 = SelectionLayer(name='s6')(inputs)
s7 = SelectionLayer(name='s7')(inputs)
s8 = SelectionLayer(name='s8')(inputs)
v1 = ViewLayer(view_index=1, name='v1')(s1)
v2 = ViewLayer(view_index=2, name='v2')([s1, s2])
v3 = ViewLayer(view_index=3, name='v3')([s1, s2, s3])
v4 = ViewLayer(view_index=4, name='v4')([s1, s2, s3, s4])
v5 = ViewLayer(view_index=5, name='v5')([s1, s2, s3, s4, s5])
v6 = ViewLayer(view_index=6, name='v6')([s1, s2, s3, s4, s5, s6])
v7 = ViewLayer(view_index=7, name='v7')([s1, s2, s3, s4, s5, s6, s7])
v8 = ViewLayer(view_index='Last', name='v8')(s8)
concatenation = keras.layers.concatenate(
    [v1, v2, v3, v4, v5, v6, v7, v8], name='concatenation')
fully_connected = keras.layers.Dense(
    units=hidden_units, name='fully_connected')(concatenation)
dropout = keras.layers.Dropout(rate=dropout_rate)(fully_connected)
another_dense_layer = keras.layers.Dense(
    units=hidden_units, name='another_dense_layer')(dropout)
softmax = keras.layers.Dense(
    units=output_units, activation='softmax',
    name='softmax')(dropout)

model = keras.models.Model(inputs=inputs, outputs=softmax)
```

# Utilities

The `utils.py` module contains a couple of functions that could come in handy when pre-processing your input. As mentioned above, **it's important that when you coerce your list of embedded_documents to `np.array()` all the documents have a same number of embedded_tokens**. Otherwise, the resulting array will have an incorrect `.shape`, which would cause [Keras](https://keras.io/) to throw an error (as the input wouldn't match the expected shape).

There are two utility functions you can use to solve this problem: pad_embedded_corpus() and cap_embedded_corpus(). The first one adds 0-filled mock embedded_tokens to each document until all documents have the same length. The second one crops each document so that only the first X tokens are maintained, achieving the same result.

For example:

```python
import multi_view_network

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

padded_corpus = multi_view_network.pad_embedded_corpus(embedded_corpus, embeddings_dim=2)
padded_corpus_sizes = [len(lst) for lst in padded_corpus]
# padded_corps_sizes
# >>> [3, 3, 3]

capped_corpus = multi_view_network.cap_embedded_corpus(embedded_corpus)
capped_corpus_sizes = [len(lst) for lst in capped_corpus]
#capped_corpus_sizes
# >>> [1, 1, 1]
```

Adding 0-filled vectors to the documents has no effect on the output and training performance of the MVN, and it's thus the recommended way to make sure all embedded_documents have the same length.
