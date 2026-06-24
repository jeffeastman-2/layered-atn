import numpy as np
from latn.lexer.vector_space import VectorSpace
from latn.pos.conjunction_phrase import ConjunctionPhrase
from latn.utils.debug import debug_print

class SentencePhrase():
    def __init__(self):
        self.vector = VectorSpace()
        self.subject = None
        self.predicate = None
        self.definition_word = None
        self.tobe = None
        self.scale_factor = 1.0


    def to_vector(self):
        # Optional: vector for the entire sentence
        return self.predicate.to_vector()

    def __repr__(self):
        return f"Sentence(subject={self.subject}, predicate={self.predicate})"

    def store_definition_word(self, tok):
        self.definition_word = tok.word
        self.vector = tok  # Store the vector representation of the quoted word

    def apply_subject_token(self, token):
        self.apply_subject(token.phrase)

    def apply_subject(self, subj):
        debug_print(f"✅ => Applying sentence subject {subj} \n      to {self}")
        if self.subject is None:
            self.subject = subj
        elif self.subject == subj:
            return
        elif isinstance(self.subject, ConjunctionPhrase):
            self.subject.phrases.append(subj)
 
    def apply_predicate_token(self, token):
        self.apply_predicate(token.phrase)

    def apply_predicate(self, pred):
        debug_print(f"✅ => Applying sentence predicate {pred} \n      to {self}")
        if self.predicate is None:
            self.predicate = pred
        elif pred == self.predicate:
            # Avoid wrapping the same predicate twice
            return
        elif isinstance(self.predicate, ConjunctionPhrase):
            self.predicate.phrases.append(pred)

    def apply_prepositional_phrase(self, pp_token):
        """Apply a PP token to the sentence structure."""
        if not hasattr(self, 'prepositional_phrases'):
            self.prepositional_phrases = []
        self.prepositional_phrases.append(pp_token)
        return True

    def apply_tobe(self, tok):
        debug_print(f"✅ => Applying sentence tobe {tok}")
        self.tobe = tok.word
        self.vector += tok

    def apply_adverb(self, tok):
        """Store the adverb vector for use in scaling the next adjective."""
        self.scale_vector = getattr(self, "scale_vector", VectorSpace())
        debug_print(f"Scale_vector is {self.scale_vector} for token {tok}")
        self.scale_vector += tok  # Combine adverbs if needed (e.g., "very extremely")

    def apply_adjective(self, tok):
        """Apply the adverb-scaled adjective to the NP vector."""
        scale = getattr(self, "scale_vector", None)
        if scale:
            strength = scale.scalar_projection("adv")
            debug_print(f"Scaling adjective {tok.word} by {strength}")
            debug_print(f"Adjective vector before scale: {tok}")
            self.vector += tok * strength
            self.scale_vector = None
        else:
            self.vector += tok

    def equals(self, other):
        """Deep equality comparison for SentencePhrase objects."""
        if not isinstance(other, SentencePhrase):
            return False        
        if self.subject and other.subject:
            if not self.subject.equals(other.subject):
                return False            
        if self.predicate and other.predicate:
            if not self.predicate.equals(other.predicate):
                return False
        if len(self.preps) != len(other.preps):
            return False
        for this_prep, other_prep in zip(self.preps, other.preps):
            if not this_prep.equals(other_prep):
                return False
        return (
            getattr(self, 'definition_word', None) == getattr(other, 'definition_word', None) and
            getattr(self, 'tobe', None) == getattr(other, 'tobe', None) and
            getattr(self, 'scale_factor', 1.0) == getattr(other, 'scale_factor', 1.0) 
        )


    def printString(self):
        """Return a string representation of the sentence.
        
        Delegates to subject and predicate's printString() methods for
        Layer-6-friendly formatting with grounded object IDs preserved.
        """
        subject_str = self.subject.printString() if self.subject else None
        predicate_str = self.predicate.printString() if self.predicate else "[No Predicate]"
        if subject_str:
            return f"{subject_str} {predicate_str}"
        else:
            return f"{predicate_str}"

    def evaluate_boolean_function(self, func):
        return func(self)

    def descriptive_word(self):
        return f"SP(sentence)"

    @staticmethod
    def phrase_type():
        return "SP"