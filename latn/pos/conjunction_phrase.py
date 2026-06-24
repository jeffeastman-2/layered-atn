

from typing import List

class ConjunctionPhrase:
    def __init__(self, tok, phrases=None):
        self.conjunction = tok.word
        self.vector = tok
        self.phrases: List = phrases if phrases is not None else []  # NPs or PPs or VPs or SPs

    def __repr__(self):
        return f"ConjunctionPhrase({self.conjunction} {self.phrases})"

    def get_last(self):
        return self.phrases[-1] if self.phrases else None

    def printString(self):
        """Print the string representation of the conjunction phrase."""
        parts = [f"{item.printString()}" for item in self.phrases]
        str =  "{" + f" *{self.conjunction}* ".join(parts) + "}"
        if self.phrases and hasattr(self.phrases[0], 'prepositions') and self.phrases[0].prepositions:
            str = '(' + str # Add prepositions of the first (noun) phrase because they apply to the whole conjunction
            str += " " + " ".join(prep.printString() for prep in self.phrases[0].prepositions) + ")"
        return str
    
    def equals(self, other):
        """Deep equality comparison using phrases to compare tree structures."""
        if not isinstance(other, ConjunctionPhrase):
            return False
        self_phrases = self.phrases
        other_phrases = other.phrases
        # Compare the phrases lists
        if len(self_phrases) != len(other_phrases):
            return False
        for i in range(len(self_phrases)):
            if not self_phrases[i].equals(other_phrases[i]):  # Complete the comparison
                return False            
        return True

    def evaluate_boolean_function(self, func):
        if self.vector.isa("conj"):
            for phrase in self.phrases:
                    if not func(phrase):
                        return False
            return True
        elif self.vector.isa("disj"):
            for phrase in self.phrases:
                if func(phrase):
                    return True
            return False
        else:
            raise ValueError(f"Unknown conjunction type: {self.vector}")

    @property
    def grounding(self):
        """Return the grounding information for the conjunction phrase."""
        all_grounded = True
        all_objects = set()
        for phrase in self.phrases:
            if hasattr(phrase, 'grounding') and phrase.grounding:
                if 'scene_objects' in phrase.grounding:
                    all_objects.update(phrase.grounding['scene_objects'])
            else:
                all_grounded = False
        if all_grounded and all_objects:
            return {'scene_objects': all_objects}
        else:
            return None  
        
    def add_prepositional_phrase(self, phrase):
        for np in self.phrases:
            if hasattr(np, 'add_prepositional_phrase'):
                np.add_prepositional_phrase(phrase)

    @property
    def prepositions(self):
        if self.phrases and hasattr(self.phrases[0], 'prepositions'):
            return self.phrases[0].prepositions
        else:
            return []