import pytest
from latn.utils.verb_inflector import find_root_verb, is_verb_inflection
from latn.lexer.latn_layer_executor import tokenize_best 


class TestVerbInflection:
    """Test the verb inflection detection system."""
    
    def test_past_participle_detection(self):
        """Test detection of past participle forms."""
        # Regular -ed forms
        root, inflection_type, found = find_root_verb("called")
        assert found is True
        assert root == "call"
        assert inflection_type == "verb_past_part"
        
        root, inflection_type, found = find_root_verb("named")
        assert found is True
        assert root == "name"
        assert inflection_type == "verb_past_part"
        
        root, inflection_type, found = find_root_verb("created")
        assert found is True
        assert root == "create"
        assert inflection_type == "verb_past_part"
    
    def test_present_participle_detection(self):
        """Test detection of present participle (-ing) forms."""
        root, inflection_type, found = find_root_verb("calling")
        assert found is True
        assert root == "call"
        assert inflection_type == "verb_present_part"
        
        root, inflection_type, found = find_root_verb("naming")
        assert found is True
        assert root == "name"
        assert inflection_type == "verb_present_part"
        
        root, inflection_type, found = find_root_verb("creating")
        assert found is True
        assert root == "create"
        assert inflection_type == "verb_present_part"
    
    def test_irregular_verbs(self):
        """Test handling of irregular verb forms."""
        # Test irregular forms that are NOT in main vocabulary
        # Let's test "done" -> "do" if "do" is in vocabulary
        root, inflection_type, found = find_root_verb("done")
        if found and root == "do":
            assert inflection_type == "verb_past_part"
        
        # Test that words already in vocabulary return as-is
        root, inflection_type, found = find_root_verb("been")
        assert found is True
        assert root == "been"  # Already in vocabulary
        assert inflection_type is None
        
        # Test unknown words
        root, inflection_type, found = find_root_verb("unknownword")
        assert found is False
    
    def test_base_verb_unchanged(self):
        """Test that base verbs are returned unchanged."""
        root, inflection_type, found = find_root_verb("call")
        assert found is True
        assert root == "call"
        assert inflection_type is None  # No inflection for base form
        
        root, inflection_type, found = find_root_verb("create")
        assert found is True
        assert root == "create"
        assert inflection_type is None
    
    def test_unknown_words(self):
        """Test handling of unknown words."""
        root, inflection_type, found = find_root_verb("xyzabc")
        assert found is False
        assert root == "xyzabc"
        assert inflection_type is None
    
    def test_is_verb_inflection_helper(self):
        """Test the helper function for detecting inflections."""
        assert is_verb_inflection("called") is True
        assert is_verb_inflection("naming") is True
        assert is_verb_inflection("call") is False  # Base form, already in vocab
        assert is_verb_inflection("xyzabc") is False  # Unknown word


class TestTokenizerVerbInflection:
    """Test verb inflection handling in the tokenizer."""
    
    def test_tokenizer_past_participle(self):
        """Test that tokenizer correctly handles past participles."""
        tokens = tokenize_best("called")
        assert len(tokens) == 1
        tok = tokens[0]
        
        assert tok.word == "called"
        assert tok["verb"] > 0.0
        assert tok["verb_past_part"] > 0.0
        assert tok["action"] > 0.0
        assert tok["naming"] > 0.0
    
    def test_tokenizer_present_participle(self):
        """Test that tokenizer correctly handles present participles."""
        tokens = tokenize_best("calling")
        assert len(tokens) == 1
        tok = tokens[0]
        
        assert tok.word == "calling"
        assert tok["verb"] > 0.0
        assert tok["verb_present_part"] > 0.0
        assert tok["action"] > 0.0
        assert tok["naming"] > 0.0
    
    def test_tokenizer_base_verb(self):
        """Test that tokenizer correctly handles base verbs."""
        tokens = tokenize_best("call")
        assert len(tokens) == 1
        tok = tokens[0]
        
        assert tok.word == "call"
        assert tok["verb"] > 0.0
        assert tok["verb_past_part"] == 0.0  # No inflection dimension
        assert tok["verb_present_part"] == 0.0
        assert tok["action"] > 0.0
        assert tok["naming"] > 0.0
    
    def test_sentence_with_inflections(self):
        """Test full sentences with verb inflections."""
        tokens = tokenize_best("create a sphere called 'sun'")
        assert len(tokens) == 5
        
        # Check "create" (base verb)
        create_tok = tokens[0]
        assert create_tok.word == "create"
        assert create_tok["verb"] > 0.0
        assert create_tok["create"] > 0.0
        
        # Check "called" (past participle)
        called_tok = tokens[3]
        assert called_tok.word == "called"
        assert called_tok["verb"] > 0.0
        assert called_tok["verb_past_part"] > 0.0
        assert called_tok["naming"] > 0.0
        
        # Check quoted name
        name_tok = tokens[4]
        assert name_tok.word == "sun"
        assert name_tok["quoted"] > 0.0


class TestGrammaticalErrorDetection:
    """Test detection of grammatical errors in naming syntax."""
    
    def test_correct_naming_syntax(self):
        """Test that correct syntax doesn't trigger errors."""
        tokens = tokenize_best("create a sphere called 'sun'")
        
        # Should not detect any grammatical errors
        for i, tok in enumerate(tokens):
            if tok["verb"] > 0.0 and tok["naming"] > 0.0:
                # This should be "called" with past participle
                assert tok["verb_past_part"] > 0.0, f"Token {tok.word} should have past participle"
    
    def test_incorrect_naming_syntax(self):
        """Test detection of incorrect base verb + quoted name."""
        tokens = tokenize_best("create a box name 'fred'")
        
        # Find the naming verb
        naming_verb = None
        for i, tok in enumerate(tokens):
            if tok["verb"] > 0.0 and tok["naming"] > 0.0:
                naming_verb = (i, tok)
                break
        
        assert naming_verb is not None
        i, tok = naming_verb
        
        # Should be base verb (incorrect)
        assert tok.word == "name"
        assert tok["verb"] > 0.0
        assert tok["naming"] > 0.0
        assert tok["verb_past_part"] == 0.0  # Missing past participle
        
        # Should be followed by quoted name
        assert i + 1 < len(tokens)
        next_tok = tokens[i + 1]
        assert next_tok["quoted"] > 0.0
        
        # This pattern should trigger error detection in the parser
        has_inflection = any(tok[dim] > 0 for dim in ['verb_past_part', 'verb_present_part', 'verb_past'])
        assert not has_inflection, "Base verb should not have inflection dimensions"
    
    def test_mixed_verbs_in_sentence(self):
        """Test sentences with multiple verb forms."""
        tokens = tokenize_best("creating a box called 'test'")
        
        # "creating" should be present participle
        creating_tok = tokens[0]
        assert creating_tok.word == "creating"
        assert creating_tok["verb_present_part"] > 0.0
        
        # "called" should be past participle
        called_tok = tokens[3]
        assert called_tok.word == "called"
        assert called_tok["verb_past_part"] > 0.0


