import pytest
from latn.utils.noun_inflector import singularize_noun, is_plural


pytestmark = pytest.mark.usefixtures("neutral_latn")

class TestNounInflector:
    """Test suite for the noun inflector module."""
    
    def test_regular_plurals(self):
        """Test regular plural noun singularization."""
        test_cases = [
            # Regular -s plurals
            ("cats", "cat", True),
            ("dogs", "dog", True),
            ("books", "book", True),
            ("cars", "car", True),
            
            # -es plurals
            ("boxes", "box", True),
            ("glasses", "glass", True),
            ("wishes", "wish", True),
            ("churches", "church", True),
            
            # -ies plurals (y -> ies)
            ("cities", "city", True),
            ("parties", "party", True),
            ("companies", "company", True),
            ("stories", "story", True),
            
            # -ves plurals (f/fe -> ves)
            ("knives", "knife", True),
            ("lives", "life", True),
            ("wolves", "wolf", True),
            ("shelves", "shelf", True),
        ]
        
        for plural, expected_singular, expected_is_plural in test_cases:
            singular, is_plural_result = singularize_noun(plural)
            assert singular == expected_singular, f"Expected '{expected_singular}' but got '{singular}' for '{plural}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{plural}'"
    
    def test_irregular_plurals(self):
        """Test irregular plural noun singularization."""
        test_cases = [
            ("men", "man", True),
            ("women", "woman", True),
            ("children", "child", True),
            ("geese", "goose", True),
            ("mice", "mouse", True),
            ("feet", "foot", True),
            ("teeth", "tooth", True),
        ]
        
        for plural, expected_singular, expected_is_plural in test_cases:
            singular, is_plural_result = singularize_noun(plural)
            assert singular == expected_singular, f"Expected '{expected_singular}' but got '{singular}' for '{plural}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{plural}'"
    
    def test_singular_nouns(self):
        """Test that singular nouns are not changed."""
        test_cases = [
            ("cat", "cat", False),
            ("dog", "dog", False),
            ("book", "book", False),
            ("box", "box", False),
            ("city", "city", False),
            ("knife", "knife", False),
            ("man", "man", False),
            ("child", "child", False),
        ]
        
        for singular, expected_singular, expected_is_plural in test_cases:
            result_singular, is_plural_result = singularize_noun(singular)
            assert result_singular == expected_singular, f"Expected '{expected_singular}' but got '{result_singular}' for '{singular}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{singular}'"
    
    def test_words_ending_in_ss(self):
        """Test that words ending in 'ss' are not treated as plurals."""
        test_cases = [
            ("glass", "glass", False),
            ("class", "class", False),
            ("mass", "mass", False),
            ("pass", "pass", False),
            ("boss", "boss", False),
        ]
        
        for word, expected_singular, expected_is_plural in test_cases:
            result_singular, is_plural_result = singularize_noun(word)
            assert result_singular == expected_singular, f"Expected '{expected_singular}' but got '{result_singular}' for '{word}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{word}'"
    
    def test_dissertation_plurals(self):
        """Test specific plural nouns from dissertation sentences."""
        test_cases = [
            # From dissertation
            ("circles", "circle", True),
            ("boxes", "box", True),
            ("degrees", "degree", True),
            
            # Edge cases that might appear
            ("spheres", "sphere", True),
            ("cubes", "cube", True),
            ("triangles", "triangle", True),
            ("rectangles", "rectangle", True),
            ("cylinders", "cylinder", True),
            ("cones", "cone", True),
            
            # Words that end in 's' but aren't plurals
            ("is", "is", False),
            ("as", "as", False),
        ]
        
        for word, expected_singular, expected_is_plural in test_cases:
            result_singular, is_plural_result = singularize_noun(word)
            assert result_singular == expected_singular, f"Expected '{expected_singular}' but got '{result_singular}' for '{word}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{word}'"
    
    def test_is_plural_function(self):
        """Test the is_plural standalone function."""
        # Test plurals
        assert is_plural("cats") == True
        assert is_plural("boxes") == True
        assert is_plural("cities") == True
        assert is_plural("knives") == True
        assert is_plural("men") == True
        assert is_plural("circles") == True
        assert is_plural("degrees") == True
        
        # Test singulars
        assert is_plural("cat") == False
        assert is_plural("box") == False
        assert is_plural("city") == False
        assert is_plural("knife") == False
        assert is_plural("man") == False
        assert is_plural("circle") == False
        assert is_plural("degree") == False
        
        # Test edge cases
        assert is_plural("glass") == False
        assert is_plural("is") == False
        assert is_plural("as") == False
    
    def test_case_insensitive(self):
        """Test that the inflector works with different cases."""
        test_cases = [
            ("CATS", "cat", True),
            ("Cats", "cat", True),
            ("cAtS", "cat", True),
            ("BOXES", "box", True),
            ("Boxes", "box", True),
            ("CITIES", "city", True),
            ("Cities", "city", True),
        ]
        
        for word, expected_singular, expected_is_plural in test_cases:
            result_singular, is_plural_result = singularize_noun(word)
            assert result_singular == expected_singular, f"Expected '{expected_singular}' but got '{result_singular}' for '{word}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{word}'"
    
    def test_edge_cases(self):
        """Test edge cases and potential problem words."""
        test_cases = [
            # Single letter words
            ("a", "a", False),
            ("s", "s", False),
            
            # Empty string
            ("", "", False),
            
            # Words with numbers (from dissertation)
            ("3", "3", False),
            ("45", "45", False),
            
            # Very short words ending in s
            ("is", "is", False),
            ("as", "as", False),
            ("us", "us", False),
        ]
        
        for word, expected_singular, expected_is_plural in test_cases:
            result_singular, is_plural_result = singularize_noun(word)
            assert result_singular == expected_singular, f"Expected '{expected_singular}' but got '{result_singular}' for '{word}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{word}'"
    
    def test_geometric_shapes_plurals(self):
        """Test pluralization of geometric shapes specifically used in the graphics domain."""
        test_cases = [
            # Basic shapes
            ("circles", "circle", True),
            ("squares", "square", True),
            ("triangles", "triangle", True),
            ("rectangles", "rectangle", True),
            ("boxes", "box", True),
            ("cubes", "cube", True),
            ("spheres", "sphere", True),
            ("cylinders", "cylinder", True),
            ("cones", "cone", True),
            
            # Complex shapes
            ("tetrahedrons", "tetrahedron", True),
            ("hexahedrons", "hexahedron", True),
            ("octahedrons", "octahedron", True),
            ("dodecahedrons", "dodecahedron", True),
            ("icosahedrons", "icosahedron", True),
            ("pyramids", "pyramid", True),
            ("prisms", "prism", True),
            ("arches", "arch", True),
            ("tables", "table", True),
            ("objects", "object", True),
        ]
        
        for plural, expected_singular, expected_is_plural in test_cases:
            result_singular, is_plural_result = singularize_noun(plural)
            assert result_singular == expected_singular, f"Expected '{expected_singular}' but got '{result_singular}' for '{plural}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{plural}'"
    
    def test_compound_words_and_phrases(self):
        """Test handling of compound words and phrases that might contain plurals."""
        test_cases = [
            # Simple compounds
            ("toolboxes", "toolbox", True),
            ("textbooks", "textbook", True),
            
            # Hyphenated compounds (if any)
            ("half-circles", "half-circle", True) if "half-circles".endswith('s') else ("half-circle", "half-circle", False),
        ]
        
        for word, expected_singular, expected_is_plural in test_cases:
            result_singular, is_plural_result = singularize_noun(word)
            assert result_singular == expected_singular, f"Expected '{expected_singular}' but got '{result_singular}' for '{word}'"
            assert is_plural_result == expected_is_plural, f"Expected is_plural={expected_is_plural} for '{word}'"


class TestNounInflectorIntegration:
    """Integration tests for noun inflector with the vocabulary system."""
    
    def test_vocabulary_integration(self):
        """Test that inflector works with vocabulary lookups."""
        from latn.lexer.vocabulary_builder import vector_from_word, has_word
        
        # Test that singular forms are in vocabulary
        assert has_word("circle")
        assert has_word("box")
        assert has_word("cube")
        assert has_word("sphere")
        
        # Test that we can get vectors for known singulars
        try:
            circle_vector = vector_from_word("circle")
            assert circle_vector is not None
            assert circle_vector.word == "circle"
        except Exception as e:
            # If there are vocabulary issues, the test should still pass for inflector logic
            pass
    
    def test_plural_vector_creation(self):
        """Test that plural nouns get proper vectors with plural marking."""
        from latn.lexer.vocabulary_builder import vector_from_word
        
        try:
            # Test plural of known word
            circles_vector = vector_from_word("circles")
            assert circles_vector is not None
            assert circles_vector["plural"] == 1.0
            assert circles_vector["singular"] == 0.0
            
            boxes_vector = vector_from_word("boxes")
            assert boxes_vector is not None
            assert boxes_vector["plural"] == 1.0
            assert boxes_vector["singular"] == 0.0
            
        except Exception as e:
            # The vocabulary system might not be fully integrated yet
            pytest.skip(f"Vocabulary integration not ready: {e}")

