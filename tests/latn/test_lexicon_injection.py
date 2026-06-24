"""Contract test: the LATN tokenizer resolves words through the *active*
Lexicon, not a hardcoded Engraf global.

Engraf's 261 latn tests prove no regression of the default (Engraf) lexicon.
They do not prove the lexicon is injectable. This feeds the tokenizer a
custom lexicon containing a nonsense word absent from Engraf's vocabulary
and absent from every morphology fallback, proving (a) injection works and
(b) `use_lexicon` scoping does not leak into the rest of the suite. This is
the precondition for Driftmoor supplying an RPG vocabulary in phase 2.
"""

from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.lexer.lexicon import (
    Lexicon,
    get_active_lexicon,
    set_active_lexicon,
    use_lexicon,
)
from latn.lexer.vector_space import vector_from_features

# Not in Engraf's vocabulary; not -s/-ed/-ing/-er so no morphology fallback.
NONSENSE = "zorblax"


def _first_token(sentence):
    result = LATNLayerExecutor().execute_layer1(sentence)
    assert result.success and result.hypotheses
    return result.hypotheses[0].tokens[0]


def test_lexicon_is_a_dict_facade():
    lex = Lexicon({})
    vs = vector_from_features("noun")
    lex["foo"] = vs
    assert "foo" in lex
    assert lex["foo"] is vs
    assert lex.get("bar") is None
    assert len(lex) == 1


def test_default_lexicon_wraps_engraf_vocabulary():
    assert "cube" in get_active_lexicon()


def test_injected_lexicon_is_used_and_scoped():
    custom = Lexicon({NONSENSE: vector_from_features("noun")})

    # Baseline: unknown under the default Engraf lexicon.
    assert _first_token(NONSENSE).isa("unknown")

    # Injected: the same word now parses with the custom POS.
    with use_lexicon(custom):
        tok = _first_token(NONSENSE)
        assert tok.isa("noun")
        assert not tok.isa("unknown")

    # Restored: no leak back into the default lexicon.
    assert _first_token(NONSENSE).isa("unknown")


def test_set_active_lexicon_reset():
    custom = Lexicon({NONSENSE: vector_from_features("noun")})
    try:
        set_active_lexicon(custom)
        assert _first_token(NONSENSE).isa("noun")
    finally:
        set_active_lexicon(None)  # reset to Engraf default
    assert _first_token(NONSENSE).isa("unknown")
    assert "cube" in get_active_lexicon()
