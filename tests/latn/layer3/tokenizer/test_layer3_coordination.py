import pytest

from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.pos.conjunction_phrase import ConjunctionPhrase
from latn.pos.noun_phrase import NounPhrase
from latn.pos.prepositional_phrase import PrepositionalPhrase


pytestmark = pytest.mark.usefixtures("neutral_latn")



def test_coordinated_pp():
    """Test Layer 3 PP tokenization with coordinated prepositional phrases """
    executor = LATNLayerExecutor()

    # Test coordinated PPs: "above the red box and below the blue circle and behind the octahedron"
    result = executor.execute_layer3('above the red box and below the blue circle and behind the octahedron',tokenize_only=True, report=True)

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) >= 1, "Should generate 1 hypothesis"

    main_hyp = result.hypotheses[0]
    # Should have exactly 3 PP tokens (one for each prepositional phrase)
    assert len(main_hyp.tokens) == 1, f"Should have exactly 1 token, got {len(main_hyp.tokens)}"
    main_pp = main_hyp.tokens[0]
    assert main_pp.word == "CONJ-PP", "First token should be a CONJ-PP"
    parts = list(main_pp.phrase.phrases)
    assert len(parts) == 3, f"CONJ-PP should have 3 parts, got {len(parts)}"
    assert all(isinstance(part, PrepositionalPhrase) for part in parts), "All parts should be PrepositionalPhrase instances"
    assert parts[0].preposition == "above", f"First PP should be 'above', got '{parts[0].preposition}'"
    assert parts[1].preposition == "below", f"Second PP should be 'below', got '{parts[1].preposition}'"
    assert parts[2].preposition == "behind", f"Third PP should be 'behind', got '{parts[2].preposition}'"

def test_coordinated_pp_with_nps():

    executor = LATNLayerExecutor()

    result = executor.execute_layer3('the red cube above the table and the blue sphere below the cylinder',tokenize_only=True, report=True)

    assert result.success, "Failed to tokenize coordinated PPs in Layer 3"
    assert len(result.hypotheses) >= 2, "Should generate 2 hypotheses"

    hyp = result.hypotheses[4]
    assert len(hyp.tokens) >= 1, f"Should have exactly 1 token, got {len(hyp.tokens)}"
    np = hyp.tokens[0].phrase
    str = np.printString()
    assert str == "the red cube (above {the table *and* the blue sphere} below the cylinder)", f"First token should be NP, got {hyp.tokens[0].word}"
