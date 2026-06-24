# LATN Branch: Using multi-hypothesis tokenization
# Original single-hypothesis tokenizer preserved on main branch

import re
import warnings
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_best, latn_tokenize_layer1, TokenizationHypothesis
from latn.lexer.vector_space import VectorSpace


class TokenStream:
    def __init__(self, tokens: list[VectorSpace]):
        self.tokens = tokens
        self.position = 0

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]

    def __repr__(self):
        return f"TokenStream(pos={self.position}, total={len(self.tokens)})"

    def peek(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def next(self):
        if self.position < len(self.tokens):
            tok = self.tokens[self.position]
            self.position += 1
            return tok
        return None

    def reset(self):
        self.position = 0

    def advance(self):
        if self.position < len(self.tokens):
            self.position += 1
        else:
            raise IndexError("TokenStream position out of bounds")


def tokenize(sentence):
    """
    DEPRECATED: Use latn_tokenize() or latn_tokenize_best() instead.
    
    LATN Branch: Multi-hypothesis tokenization with best hypothesis fallback.
    
    For backward compatibility, this returns the best single hypothesis.
    Use latn_tokenize() directly for full multi-hypothesis analysis.
    """
    warnings.warn(
        "tokenize() is deprecated. Use latn_tokenize() for multi-hypothesis analysis "
        "or latn_tokenize_best() for single best hypothesis.",
        DeprecationWarning,
        stacklevel=2
    )
    return latn_tokenize_best(sentence)


# Export multi-hypothesis functionality
__all__ = ['TokenStream', 'tokenize', 'latn_tokenize_layer1', 'TokenizationHypothesis']
