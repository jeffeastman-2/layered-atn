
from latn.lexer.vector_space import VectorSpace
from latn.utils.debug import debug_print

class VerbPhrase():
    def __init__(self):
        # VerbPhrase doesn't need to call super() if it doesn't inherit from anything
        self.verb = None
        self.vector = VectorSpace()
        self.noun_phrase = None
        self.prepositions = []
        self.adjective_complements = []  # Changed to list for multiple adjectives
        self.amount = None  # For handling measurements like "45 degrees"

    def __repr__(self):
        return f"VerbPhrase(verb={self.verb}, noun_phrase={self.noun_phrase}, PPs={self.prepositions}, adjective_complement={self.adjective_complements}, amount={self.amount})"

    def to_vector(self) -> VectorSpace:
        # Combine verb meaning with its object’s vector (if present)
        v = self.verb.to_vector().copy()
        if self.noun_phrase:
            v += self.noun_phrase.to_vector()
        return v

    def apply_verb(self, tok):
        if self.verb is None:
            self.verb = tok.word
        else:
            self.verb = (f"{self.verb} {tok.word}")
        self.vector += tok

    def apply_adverb(self, tok):
        if self.verb is None:
            self.verb = tok.word
        else:
            self.verb = (f"{self.verb} {tok.word}")

    def apply_conjunction(self, tok):
        self.verb = (f"{self.verb} {tok.word}")

    def apply_negation(self, tok):
        if self.verb is None:
            self.verb = tok.word
        else:
            self.verb = (f"{tok.word} {self.verb}")
        self.vector += tok

    def apply_np(self, np):
        self.noun_phrase = np

    def apply_amount(self, amount_np):
        """Apply an amount/measure noun phrase like '45 degrees'"""
        self.amount = amount_np
        debug_print(f"✅ VP applying amount: {amount_np}")

    def apply_pp(self, pp_obj):
        debug_print(f"✅ VP applying PP: {pp_obj}")
        self.prepositions.append(pp_obj)

    def apply_adjective(self, tok):
        self.adjective_complements.append(tok.word)
        debug_print(f"✅ VP applying adjective complement: {tok.word}")
        debug_print(f"✅ Current adjective complements: {self.adjective_complements}")

    def is_imperative(self):
        return self.vector.scalar_projection("action") > 0.5

    def verb_has_intent(self, intent: str, threshold=0.5) -> bool:
        return self.verb and self.verb.scalar_projection(intent) > threshold

    def printString(self):
        """Return a string representation of the verb phrase.
        
        Delegates to child phrases' printString() methods for Layer-6-friendly
        formatting. Properly handles grounded noun phrases with object IDs.
        """
        out = self.verb or "<verb>"
        if self.noun_phrase:
            out += f" [{self.noun_phrase.printString()}]"
        if self.prepositions:
            out += " -> ("
            for pp in self.prepositions:
                out += f"({pp.printString()})"
            out += ")"
        if self.adjective_complements:
            adjectives = " ".join(self.adjective_complements)
            out += f" => ({adjectives})"
        if self.amount:
            out += f" == [{self.amount.printString()}]"
        return out

    def equals(self, other):
        """Deep equality comparison for VerbPhrase objects."""
        if not isinstance(other, VerbPhrase):
            return False
        if len(self.prepositions) != len(other.prepositions):
            return False
        for this_prep, other_prep in zip(self.prepositions, other.prepositions):
            if not this_prep.equals(other_prep):
                return False  
        if len(self.adjective_complements) != len(other.adjective_complements):
            return False
        for this_adj, other_adj in zip(self.adjective_complements, other.adjective_complements):
            if this_adj != other_adj:
                return False
        return (
            self.verb == other.verb and
            self.noun_phrase.equals(other.noun_phrase) and
            self.amount == other.amount and
            getattr(self, 'vector', None) == getattr(other, 'vector', None)
        )

    def evaluate_boolean_function(self, func):
        return func(self)

    def descriptive_word(self):
        return f"VP({self.verb})"

    @staticmethod
    def phrase_type():
        return "VP"