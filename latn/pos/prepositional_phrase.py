from latn.lexer.vector_space import VectorSpace   
 

class PrepositionalPhrase():
    """Represents a prepositional phrase in a sentence.
    This is a non-terminal part of speech (POS) type.
    """
    def __init__(self):
        super().__init__()
        self.preposition = None
        self.noun_phrase = None 
        self.vector = VectorSpace()
        
    def __repr__(self):
        return f"PrepositionalPhrase(preposition={self.preposition!r}, noun_phrase={self.noun_phrase!r})"

    def apply_preposition(self, tok):
        if self.preposition is None:
            self.preposition = tok.word
        else:
            self.preposition = (f"{self.preposition} {tok.word}")
        self.vector += tok
        
    def apply_negation(self, tok):
        if self.preposition is None:
            self.preposition = tok.word
        else:
            self.preposition = (f"{tok.word} {self.preposition}")
        self.vector += tok

    def apply_vector(self, tok):
        self.vector += tok

    def apply_np(self, np):
        self.noun_phrase = np

    def printString(self):
        """Return a string representation of the prepositional phrase.
        
        Delegates to the noun phrase's printString() method for Layer-6-friendly
        formatting. Falls back to vector info if no noun phrase.
        """
        if self.noun_phrase:
            return f"{self.preposition} {self.noun_phrase.printString()}"
        return f"{self.preposition} + {self.vector.non_zero_dims()}"

    def equals(self, other):
        """Deep equality comparison for PrepositionalPhrase objects."""
        if not isinstance(other, PrepositionalPhrase):
            return False        
        return (
            self.preposition == other.preposition and
            self.noun_phrase.equals(other.noun_phrase) and
            getattr(self, 'vector', None) == getattr(other, 'vector', None) and
            getattr(self, 'vector_text', None) == getattr(other, 'vector_text', None)
        )

    def evaluate_boolean_function(self, func):
        return func(self)

    def descriptive_word(self):
        return f"PP({self.preposition})"

    @staticmethod
    def phrase_type():
        return "PP"