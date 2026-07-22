
import pytest

from latn.lexer.latn_layer_executor import LATNLayerExecutor
from latn.pos.prepositional_phrase import PrepositionalPhrase


pytestmark = pytest.mark.usefixtures("neutral_latn")


class TestLayer3Negation:

    def test_interrogative_sentence_negation2(self):   
        """Test: 'not on the table' -> PP token."""
        executor = LATNLayerExecutor()
        result = executor.execute_layer3("not on the table", report=True, tokenize_only=True)

        assert result.success, "Layer 3 should succeed"
        assert len(result.hypotheses) > 0, "Should have PP hypotheses"
        
        hyp = result.hypotheses[0]
        pp = hyp.tokens[0].phrase
        assert isinstance(pp, PrepositionalPhrase), "Should be PrepositionalPhrase object"
        assert pp.preposition is not None, "Should have preposition"
        assert pp.noun_phrase is not None, "Should have noun phrase"
        assert pp.preposition == "not on", "Preposition should include negation"
        assert "table" in pp.noun_phrase.noun, "PP NP should be 'table'"
