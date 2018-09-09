"""Module containing utilities to use multi_view_network."""

import copy


def check_shape(to_check, expected):
    """Checks that shape matches expectations.

    In practice, the function simply check whether the two lists
    are the same and raise an AssertionError if not.

    Args:
        to_check: [int], the shape to check.
        expected: [int], the expected shape.
    """
    try:
        assert to_check == expected
    except AssertionError:
        message = ('Shape doesn\'t meet expecations. I\'m '
                   'afraid {to_check} != {expected}'.format(to_check=to_check,
                                                            expected=expected))
        raise AssertionError(message)


def pad_embedded_corpus(embedded_corpus, embeddings_dim):
    """Pads all embedded_documents so they have all same number of embeddings.

    For Keras to read the input's shape correctly, it's important all
    embedded_documents have the same number of embedded_token. This is
    normally not the case because documents can have differnet numbers
    of tokens to begin with. This function takes care of that. Another
    option is to use cap_embedded_corpus(), though it's only reccomended
    when there are performance issues as pad_embedded_corpus() introduces
    no noise and there's no loss of information from the corpus.

    Args:
        embedded_corpus: [[[int|float]]], the first list-structure
            stores the various documents. The second list-structure
            stores the various tokens. The third list-structure stores
            the individual embeddings values.
        embeddings_dim: int, the size of the embedding space (which is
            also the length of the inner-most lists).

    Returns:
        Same structure as embedded_corpus but now all embedded_documents
        have the same number of embedded_tokens.
    """
    # To avoid over-writing the input.
    embedded_corpus = copy.deepcopy(embedded_corpus)

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


def cap_embedded_corpus(
        embedded_corpus, cap_size=None, exclude_if_below_cap=True):
    if not cap_size:
        sizes = [len(embedded_document)
                 for embedded_document in embedded_corpus]
        cap_size = min(sizes)

    capped_corpus = []
    for embedded_document in embedded_corpus:
        if (len(embedded_document[0:cap_size]) >= cap_size
                or not exclude_if_below_cap):
            capped_corpus.append(embedded_document[0:cap_size])
    return capped_corpus
