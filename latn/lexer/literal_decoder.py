"""Host seam for converting bracketed numeric literals into vectors."""

from typing import Optional, Protocol, Sequence, runtime_checkable

from latn.lexer.vector_space import VectorSpace, vector_from_features


@runtime_checkable
class LiteralDecoder(Protocol):
    def decode(self, values: Sequence[float]) -> VectorSpace:
        ...


class CoreLiteralDecoder:
    """Preserve literal structure without assigning host semantics."""

    def decode(self, values: Sequence[float]) -> VectorSpace:
        vector = vector_from_features("literal")
        vector.data = {"values": tuple(values)}
        return vector


_active: Optional[LiteralDecoder] = None


def get_active_literal_decoder() -> LiteralDecoder:
    global _active
    if _active is None:
        _active = CoreLiteralDecoder()
    return _active


def set_active_literal_decoder(decoder: Optional[LiteralDecoder]) -> None:
    global _active
    _active = decoder


class use_literal_decoder:
    def __init__(self, decoder: LiteralDecoder):
        self._new = decoder
        self._prev = None

    def __enter__(self):
        global _active
        self._prev = _active
        _active = self._new
        return self._new

    def __exit__(self, *exc):
        global _active
        _active = self._prev


__all__ = [
    "LiteralDecoder", "CoreLiteralDecoder", "get_active_literal_decoder",
    "set_active_literal_decoder", "use_literal_decoder",
]
