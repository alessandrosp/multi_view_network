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
