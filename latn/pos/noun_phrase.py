from latn.lexer.vector_space import VectorSpace   
from latn.utils.debug import debug_print   


class NounPhrase():
    def __init__(self, noun=None):
        self.vector = VectorSpace()
        self.noun = noun
        self.pronoun = None
        self.determiner = None
        self.prepositions = []
        self.scale_factor = 1.0
        self.proper_noun = None  # For proper noun names like "called 'Charlie'"
        self.consumed_tokens = []  # All original tokens that were consumed to build this NP
        self.is_unit = False  # True if this NP represents a unit (e.g., "meters", "degrees")
        self._grounding = None        # Set in L2 grounding

    @property
    def grounding(self):
        return self._grounding

    @grounding.setter  
    def grounding(self, value):
        if value is not None and not isinstance(value, dict):
            raise TypeError(f"Grounding must be dict or None, got {type(value)}")
        self._grounding = value

    def apply_determiner(self, tok):
        self.is_unit = self.is_unit or tok.isa("unit") 
        self.determiner = tok.word
        self.vector += tok
        self.consumed_tokens.append(tok)

    def apply_vector(self, tok):
        self.noun = "vector"
        self.vector += tok
        self.consumed_tokens.append(tok)
        self.is_unit = True

    def apply_adverb(self, tok):
        """Store the adverb vector for use in scaling the next adjective."""
        if not hasattr(self, 'scale_vector') or self.scale_vector is None:
            self.scale_vector = VectorSpace()
        debug_print(f"✅ Scale_vector is {self.scale_vector} for token {tok}")
        self.scale_vector += tok  # Combine adverbs if needed (e.g., "very extremely")
        self.consumed_tokens.append(tok)

    def apply_adjective(self, tok):
        """Apply the adverb-scaled adjective to the NP vector."""
        scale = getattr(self, "scale_vector", None)
        if scale:
            strength = scale.scalar_projection("adv")
            debug_print(f"✅ Scaling adjective {tok.word} by {strength}")
            debug_print(f"✅ Adjective vector before scale: {tok}")
            self.vector += tok * strength
            self.scale_vector = None
        else:
            debug_print(f"✅ Setting adjective vector without scale: {tok}")
            self.vector += tok
        self.consumed_tokens.append(tok)

    def apply_noun(self, tok):
        debug_print(f"✅ NP applying noun {tok.word} with vector {tok}")
        
        # Check number agreement between determiner and noun using vector dimensions
        if self.determiner and hasattr(self, 'vector') and self.vector:
            # Get number information from vectors
            noun_is_plural = tok["plural"] > 0.0
            noun_is_singular = tok["singular"] > 0.0
            determiner_is_singular = self.vector["singular"] > 0.0
            determiner_number = self.vector["number"]
            
            # Check for number agreement violations using vector dimensions
            if self._has_number_agreement_error(determiner_is_singular, determiner_number, noun_is_plural, noun_is_singular):
                if determiner_is_singular and noun_is_plural:
                    error_msg = f"Number agreement error: singular determiner '{self.determiner}' cannot modify plural noun '{tok.word}'"
                elif determiner_number > 1.0 and noun_is_singular:
                    error_msg = f"Number agreement error: plural determiner '{self.determiner}' (number={determiner_number}) cannot modify singular noun '{tok.word}'"
                else:
                    error_msg = f"Number agreement error between '{self.determiner}' and '{tok.word}'"
                debug_print(f"❌ {error_msg}")
                raise ValueError(error_msg)
        
        self.is_unit = self.is_unit or tok.isa("unit") or tok.isa("vector")
        self.noun = tok.word
        self.vector += tok
        self.consumed_tokens.append(tok)
    
    def _has_number_agreement_error(self, determiner_is_singular, determiner_number, noun_is_plural, noun_is_singular):
        """Check if there's a number agreement error using vector dimensions."""
        # Determine noun number: if not explicitly marked as plural, treat as singular
        is_noun_plural = noun_is_plural > 0.0
        is_noun_singular = not is_noun_plural  # unmarked nouns are singular by default
        
        # Rule 1: Singular determiners (marked with singular=1.0) should only work with singular nouns
        if determiner_is_singular and is_noun_plural:
            return True
            
        # Rule 2: Numeric determiners > 1 should only work with plural nouns
        if determiner_number > 1.0 and is_noun_singular:
            return True
            
        return False

    def apply_pronoun(self, tok):
        debug_print(f"✅ NP applying pronoun {tok.word} with vector {tok}")
        self.pronoun = tok.word
        self.vector += tok
        self.consumed_tokens.append(tok)

    def apply_proper_noun(self, name_token, has_determiner=False):
        """Apply a proper noun from 'called' or 'named' syntax.
        
        Args:
            name_token: The quoted name token
            has_determiner: True if preceded by 'a/an', making it a type designation
        """
        if has_determiner:
            # "called a 'sun'" - type designation, not proper noun
            debug_print(f"✅ NP applying type designation: {name_token.word}")
            # This will be handled as a regular noun with determiner
        else:
            # "called 'Charlie'" - proper noun
            debug_print(f"✅ NP applying proper noun: {name_token.word}")
            self.proper_noun = name_token.word
        # Don't add to vector as this is a naming directive, not semantic content

    def apply_pp(self, pp_obj):
        debug_print(f"✅ Np applying PP: {pp_obj}")
        self.prepositions.append(pp_obj)
        self.vector += pp_obj.vector

    def to_vector(self):
        v = self.noun.to_vector().copy()
        if self.determiner:
            v += self.determiner.to_vector()
        for adj in self.adjectives:
            adj.modify(v)
        for prep in self.prepositions:
            v += prep.to_vector()
        return v

    def __repr__(self):
        consumed_words = [tok.word for tok in self.consumed_tokens]
        
        # If grounded, show grounding information in a more readable format
        if self._grounding and self._grounding.get('scene_object'):
            scene_obj = self._grounding['scene_object']
            confidence = self._grounding.get('confidence', 0.0)
            grounded_desc = f"grounded→{scene_obj.name}@{confidence:.2f}"
            parts = [f"noun={self.noun}", f"{grounded_desc}", f"vector={self.vector}", f"PPs={self.prepositions}", f"consumed={consumed_words}"]
        else:
            parts = [f"noun={self.noun}", f"vector={self.vector}", f"PPs={self.prepositions}", f"consumed={consumed_words}"]
        
        return f"NP({', '.join(parts)})"
    
    def get_consumed_words(self):
        """Return list of words from consumed tokens."""
        return [tok.word for tok in self.consumed_tokens]
    
    def get_original_text(self):
        """Reconstruct the original text from consumed tokens."""
        return " ".join(self.get_consumed_words())
    
    def token_span(self):
        """Return the span (start, end) of consumed tokens for error reporting."""
        if not self.consumed_tokens:
            return (0, 0)
        # Assuming tokens have position information (would need to be added to token structure)
        return (0, len(self.consumed_tokens))  # Placeholder - would use actual positions
    
    def printString(self):
        """Return a string for logging that is Layer-6 friendly.

        If the NP is grounded to scene objects, prefer to show object IDs
        in angle brackets (e.g. <sphere_1>). Falls back to entity_id or name
        if `object_id` is not present. Otherwise returns the consumed words.
        Prepositional attachments are appended in parentheses.
        """
        out = ""
        grounding = self.grounding
        if grounding:
            # Prefer 'scene_objects' (plural) which Layer-6 structural uses
            if 'scene_objects' in grounding and grounding['scene_objects']:
                scene_objs = grounding['scene_objects']
                ids = []
                for obj in scene_objs:
                    oid = getattr(obj, 'object_id', None) or getattr(obj, 'entity_id', None) or getattr(obj, 'name', None)
                    ids.append(str(oid))
                out = f"<{ ' '.join(ids) } >".replace('  ', ' ')
            # Backwards-compatible single object key
            elif 'scene_object' in grounding and grounding['scene_object']:
                obj = grounding['scene_object']
                oid = getattr(obj, 'object_id', None) or getattr(obj, 'entity_id', None) or getattr(obj, 'name', None)
                out = f"<{oid}>"
        if not out:
            out = ' '.join(self.get_consumed_words())

        if self.prepositions:
            out = f"{out} ({' '.join(prep.printString() for prep in self.prepositions)})"
        return out
         

    def equals(self, other):
        """Deep equality comparison for NounPhrase objects."""
        if not isinstance(other, NounPhrase):
            return False
        if len(self.prepositions) != len(other.prepositions):
            return False
        for this_prep, other_prep in zip(self.prepositions, other.prepositions):
            if not this_prep.equals(other_prep):
                return False
        # Compare all simple attributes
        return (
            self.determiner == other.determiner and
            self.noun == other.noun and
            getattr(self, 'vector', None) == getattr(other, 'vector', None) and
            getattr(self, 'plural', False) == getattr(other, 'plural', False) and
            getattr(self, 'amount', None) == getattr(other, 'amount', None)
        )
    
    def evaluate_boolean_function(self, func):
        return func(self)
    
    def add_prepositional_phrase(self, prep):
        self.prepositions.append(prep)

    def descriptive_word(self):
        return f"NP({self.get_original_text()})"

    @staticmethod
    def phrase_type():
        return "NP"