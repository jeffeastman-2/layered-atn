import pytest
from latn.pos.noun_phrase import NounPhrase
from latn.lexer.latn_tokenizer_layer1 import latn_tokenize_best as tokenize
from latn.lexer.vector_space import vector_from_features


class TestProperNounHandling:
    """Test proper noun vs type designation handling in NounPhrase."""
    
    def test_proper_noun_initialization(self):
        """Test that NounPhrase initializes with proper_noun field."""
        np = NounPhrase()
        assert hasattr(np, 'proper_noun')
        assert np.proper_noun is None
    
    def test_apply_proper_noun_without_determiner(self):
        """Test applying a proper noun without determiner ('Charlie')."""
        np = NounPhrase()
        
        # Create a quoted token for the name
        tokens = tokenize("'Charlie'")
        name_token = tokens[0]
        
        # Apply as proper noun (no determiner)
        np.apply_proper_noun(name_token, has_determiner=False)
        
        assert np.proper_noun == "Charlie"
    
    def test_apply_proper_noun_with_determiner(self):
        """Test applying a type designation with determiner ('a sun')."""
        np = NounPhrase()
        
        # Create a quoted token for the name
        tokens = tokenize("'sun'")
        name_token = tokens[0]
        
        # Apply as type designation (has determiner)
        np.apply_proper_noun(name_token, has_determiner=True)
        
        # Should not set proper_noun (this is a type designation)
        assert np.proper_noun is None
    
    def test_tokenizer_proper_noun_dimension(self):
        """Test that tokenizer can create proper noun tokens."""
        # This would be used when adding proper nouns to runtime vocabulary
        from latn.lexer.vector_space import vector_from_features
        
        # Create a proper noun vector
        proper_noun_vector = vector_from_features("proper_noun")
        proper_noun_vector.word = "Charlie"
        
        assert proper_noun_vector["proper_noun"] > 0.0
        assert proper_noun_vector["noun"] == 0.0  # Should not be a common noun
    
    def test_type_designation_vector(self):
        """Test creating a type designation vector."""
        from latn.lexer.vector_space import vector_from_features
        
        # Create a type designation vector (noun + determiner)
        type_vector = vector_from_features("noun det")
        type_vector.word = "sun"
        
        assert type_vector["noun"] > 0.0
        assert type_vector["det"] > 0.0
        assert type_vector["proper_noun"] == 0.0


class TestNamingSyntaxParsing:
    """Test parsing of different naming syntax patterns."""
    
    def test_proper_noun_syntax_tokenization(self):
        """Test tokenization of proper noun naming syntax."""
        # "create a sphere called 'Charlie'"
        tokens = tokenize("create a sphere called 'Charlie'")
        
        assert len(tokens) == 5
        assert tokens[0].word == "create"  # verb
        assert tokens[1].word == "a"       # determiner
        assert tokens[2].word == "sphere"  # noun
        assert tokens[3].word == "called"  # past participle naming verb
        assert tokens[4].word == "Charlie" # quoted proper noun
        
        # Check dimensions
        assert tokens[3]["verb_past_part"] > 0.0
        assert tokens[3]["naming"] > 0.0
        assert tokens[4]["quoted"] > 0.0
    
    def test_type_designation_syntax_tokenization(self):
        """Test tokenization of type designation naming syntax."""
        # "create a sphere called a 'sun'"
        tokens = tokenize("create a sphere called a 'sun'")
        
        assert len(tokens) == 6
        assert tokens[0].word == "create"  # verb
        assert tokens[1].word == "a"       # determiner for sphere
        assert tokens[2].word == "sphere"  # noun
        assert tokens[3].word == "called"  # past participle naming verb
        assert tokens[4].word == "a"       # determiner for type designation
        assert tokens[5].word == "sun"     # quoted type name
        
        # Check dimensions
        assert tokens[3]["verb_past_part"] > 0.0
        assert tokens[3]["naming"] > 0.0
        assert tokens[4]["det"] > 0.0      # determiner
        assert tokens[5]["quoted"] > 0.0   # quoted name
    
    def test_grammatical_error_patterns(self):
        """Test patterns that should be flagged as grammatical errors."""
        # "create a box name 'fred'" - missing past participle
        tokens = tokenize("create a box name 'fred'")
        
        # Find the naming verb
        naming_verb_idx = None
        for i, tok in enumerate(tokens):
            if tok["naming"] > 0.0:
                naming_verb_idx = i
                break
        
        assert naming_verb_idx is not None
        naming_tok = tokens[naming_verb_idx]
        
        # Should be base form "name" (incorrect)
        assert naming_tok.word == "name"
        assert naming_tok["verb"] > 0.0
        assert naming_tok["verb_past_part"] == 0.0  # Missing inflection
        
        # Should be followed by quoted name
        next_tok = tokens[naming_verb_idx + 1]
        assert next_tok["quoted"] > 0.0
        
        # This pattern should be detected as an error
        is_base_verb = (naming_tok["verb"] > 0.0 and 
                       naming_tok["naming"] > 0.0 and
                       naming_tok["verb_past_part"] == 0.0)
        followed_by_quote = next_tok["quoted"] > 0.0
        
        assert is_base_verb and followed_by_quote, "Should detect error pattern"


class TestVocabularyIntegration:
    """Test integration with vocabulary system for proper nouns."""
    
    def test_proper_noun_vector_creation(self):
        """Test creating proper noun vectors for vocabulary."""
        from latn.lexer.vector_space import vector_from_features
        
        # Simulate adding "Charlie" as a proper noun to runtime vocabulary
        charlie_vector = vector_from_features("proper_noun")
        charlie_vector.word = "Charlie"
        
        assert charlie_vector["proper_noun"] > 0.0
        assert charlie_vector.word == "Charlie"
    
    def test_type_designation_vector_creation(self):
        """Test creating type designation vectors for vocabulary."""
        from latn.lexer.vector_space import vector_from_features
        
        # Simulate adding "sun" as a type designation to runtime vocabulary
        sun_vector = vector_from_features("noun det")
        sun_vector.word = "sun"
        
        assert sun_vector["noun"] > 0.0
        assert sun_vector["det"] > 0.0
        assert sun_vector.word == "sun"
    
    def test_runtime_vocabulary_lookup(self):
        """Test that proper nouns can be looked up correctly."""
        from latn.lexer.vocabulary_builder import add_to_vocabulary, get_from_vocabulary
        from latn.lexer.vector_space import vector_from_features
        
        # Add a proper noun to runtime vocabulary
        charlie_vector = vector_from_features("proper_noun")
        charlie_vector.word = "Charlie"
        add_to_vocabulary("Charlie", charlie_vector)
        
        # Look it up
        retrieved = get_from_vocabulary("Charlie")
        assert retrieved is not None
        assert retrieved["proper_noun"] > 0.0
        assert retrieved.word == "Charlie"


class TestGrammaticalConstraints:
    """Test grammatical constraints for proper nouns vs common nouns."""
    
    def test_proper_noun_no_determiner(self):
        """Test that proper nouns should not take determiners."""
        # This would be handled at the parsing level
        # "move Charlie" (correct) vs "move the Charlie" (incorrect)
        
        tokens = tokenize("Charlie")
        assert len(tokens) == 1
        # In real usage, Charlie would be in vocabulary with proper_noun dimension
        
        # Test with determiner (should be flagged as error)
        tokens = tokenize("the Charlie")
        assert len(tokens) == 2
        assert tokens[0]["det"] > 0.0  # "the"
        # Parser should detect: determiner + proper_noun = error
    
    def test_proper_noun_no_pluralization(self):
        """Test that proper nouns should not be pluralized."""
        # "Charlies" should be flagged as incorrect for proper nouns
        # This would be handled by the inflection system and parser
        
        # Proper nouns in vocabulary should not have plural forms
        from latn.lexer.vector_space import vector_from_features
        
        charlie_vector = vector_from_features("proper_noun")
        assert charlie_vector["plural"] == 0.0
        assert charlie_vector["singular"] == 0.0  # Proper nouns don't need number marking
    
    def test_type_designation_allows_articles(self):
        """Test that type designations allow determiners and pluralization."""
        from latn.lexer.vector_space import vector_from_features
        
        # Type designations should work with determiners
        sun_vector = vector_from_features("noun det")
        assert sun_vector["noun"] > 0.0
        assert sun_vector["det"] > 0.0
        
        # And should be able to pluralize: "suns"
        # (This would be handled by the noun inflection system)


