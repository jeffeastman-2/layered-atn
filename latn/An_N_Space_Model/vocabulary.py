from latn.lexer.vector_space import vector_from_features
from latn.lexer.vector_space import VectorSpace

# ---------------------------------------------------------------------------
# Function words — domain-neutral English grammar (SHARED SUBSTRATE).
#
# Closed-class words plus prepositions, degree adverbs, and punctuation. These
# carry only grammatical/relational meaning (the prepositions' spatial,
# directional, and relational dims included), never CAD content. A non-CAD
# domain (e.g. Driftmoor) builds its lexicon ON TOP of these and supplies only
# its own content words — so it reuses the preposition/spatial machinery for PP
# parsing without inheriting CAD content (cube, draw, red) that would mis-parse
# its text (e.g. "draw your sword" must not mean "create").
#
# Kept separate from _CAD_VOCABULARY; SEMANTIC_VECTOR_SPACE below is the merge
# of the two and remains Engraf's full, unchanged vocabulary.
# ---------------------------------------------------------------------------
FUNCTION_WORDS = {
    # Pronouns
    'it': vector_from_features("pronoun singular"),
    'they': vector_from_features("pronoun plural"),
    'them': vector_from_features("pronoun plural"),
    'i': vector_from_features("pronoun singular"),
    'me': vector_from_features("pronoun singular"),
    'you': vector_from_features("pronoun"),
    'we': vector_from_features("pronoun plural"),
    'us': vector_from_features("pronoun plural"),

    # Degree adverbs (intensifiers)
    'very': vector_from_features("adv", adverb=1.5),
    'more': vector_from_features("adv", adverb=1.5),
    'bright': vector_from_features("adv", adverb=1.5),
    'much': vector_from_features("adv", adverb=1.5),
    'a little bit': vector_from_features("adv", adverb=1.15),
    'extremely': vector_from_features("adv", adverb=2.0),
    'slightly': vector_from_features("adv", adverb=0.75),

    # Determiners
    'the': vector_from_features("det def"),  # Number-neutral definite article
    'one': vector_from_features("det def singular", number=1.0),
    'two': vector_from_features("det def plural", number=2.0),
    'three': vector_from_features("det def plural", number=3.0),
    'four': vector_from_features("det def plural", number=4.0),
    'five': vector_from_features("det def plural", number=5.0),
    'six': vector_from_features("det def plural", number=6.0),
    'seven': vector_from_features("det def plural", number=7.0),
    'eight': vector_from_features("det def plural", number=8.0),
    'nine': vector_from_features("det def plural", number=9.0),
    'ten': vector_from_features("det def plural", number=10.0),
    'a': vector_from_features("det singular", number=1.0),
    'an': vector_from_features("det singular", number=1.0),
    # Possessive determiners
    'my': vector_from_features("det"),
    'your': vector_from_features("det"),
    'his': vector_from_features("det"),
    'her': vector_from_features("det"),
    'its': vector_from_features("det"),
    'our': vector_from_features("det"),
    'their': vector_from_features("det"),

    # Prepositions — spatial location relationships
    'over': vector_from_features("prep spatial_location", locY=1.0),    # higher
    'above': vector_from_features("prep spatial_location", locY=1.0),   # higher
    'under': vector_from_features("prep spatial_location", locY=-1.0),  # lower
    'below': vector_from_features("prep spatial_location", locY=-1.0),  # lower
    'behind': vector_from_features("prep spatial_location", locZ=-1.0),  # behind
    'in front of': vector_from_features("prep spatial_location", locZ=1.0),  # front
    'right of': vector_from_features("prep spatial_location", locX=1.0),  # right
    'left of': vector_from_features("prep spatial_location", locX=-1.0),  # left

    # Prepositions — spatial proximity relationships
    'on': vector_from_features("prep spatial_proximity", locY=1.0),      # contact-high
    'in': vector_from_features("prep spatial_proximity", loc=[1.0, 1.0, 1.0]),     # containment
    'at': vector_from_features("prep spatial_proximity", loc=[1.0, 1.0, 1.0]),     # specific location
    'near': vector_from_features("prep spatial_proximity", loc=[1.0, 1.0, 1.0]),   # close

    # Prepositions — directional / movement
    'to': vector_from_features("prep", directional_target=1.0),    # toward destination
    'toward': vector_from_features("prep", directional_target=1.0),   # toward destination
    'towards': vector_from_features("prep", directional_target=1.0),  # toward destination
    'into': vector_from_features("prep", directional_target=1.0),     # movement into
    'onto': vector_from_features("prep", directional_target=1.0),     # movement onto
    'through': vector_from_features("prep", directional_target=1.0),  # movement via/through
    'from': vector_from_features("prep", directional_target=-1.0),  # away from source

    # Prepositions — agency / instrumentality
    'by': vector_from_features("prep", directional_agency=1.0),    # agent/means
    'with': vector_from_features("prep", directional_agency=0.7),  # accompaniment/instrument

    # Prepositions — relational
    'of': vector_from_features("prep", relational_possession=1.0),  # belongs to, part of
    'as': vector_from_features("prep", relational_comparison=1.0),  # comparison, identification
    'than': vector_from_features("prep", relational_comparison=1.0),  # comparison baseline

    # Directional adverbs that double as prepositions (movement)
    'up': vector_from_features("adv prep spatial_location", locY=1.0),       # upward
    'down': vector_from_features("adv prep spatial_location", locY=-1.0),    # downward
    'left': vector_from_features("adv prep spatial_location", locX=-1.0),    # leftward
    'right': vector_from_features("adv prep spatial_location", locX=1.0),    # rightward
    'forward': vector_from_features("adv prep spatial_location", locZ=1.0),  # forward
    'backward': vector_from_features("adv prep spatial_location", locZ=-1.0),  # backward
    'back': vector_from_features("adv prep spatial_location", locZ=-1.0),    # backward (synonym)

    # Rotational prepositions
    'around': vector_from_features("prep", directional_agency=0.8),  # rotational context
    'about': vector_from_features("prep", directional_agency=0.8),   # rotational context (synonym)

    # Conjunctions
    'and': vector_from_features("conj and"),
    'or': vector_from_features("conj or"),

    # Punctuation
    ',': vector_from_features("punct comma"),
    '.': vector_from_features("punct period"),
    '?': vector_from_features("punct question"),
    '!': vector_from_features("punct exclaim"),

    # Negation
    'not': vector_from_features("neg"),
    'no': vector_from_features("neg"),

    # Modal verbs
    'can': vector_from_features("verb modal"),
    'could': vector_from_features("verb modal"),
    'may': vector_from_features("verb modal"),
    'might': vector_from_features("verb modal"),
    'must': vector_from_features("verb modal"),
    'shall': vector_from_features("verb modal"),
    'should': vector_from_features("verb modal"),
    'will': vector_from_features("verb modal"),
    'would': vector_from_features("verb modal"),

    # wh-words
    'who': vector_from_features("wh"),
    'what': vector_from_features("wh"),
    'where': vector_from_features("wh"),
    'when': vector_from_features("wh"),
    'why': vector_from_features("wh"),
    'how': vector_from_features("wh"),
    'which': vector_from_features("wh"),

    # To be verbs
    'is': vector_from_features("tobe"),
    'are': vector_from_features("tobe"),
    'was': vector_from_features("tobe"),
    'were': vector_from_features("tobe"),
    'am': vector_from_features("tobe"),
    'be': vector_from_features("tobe"),
    'been': vector_from_features("tobe"),
    'being': vector_from_features("tobe"),
}


# ---------------------------------------------------------------------------
# CAD content vocabulary — Engraf-specific (NOT shared substrate).
#
# Shapes, measurement units, colors/sizes/textures, the CAD action verbs, the
# time-travel words, and axis references. A non-CAD domain does NOT inherit
# these; it supplies its own content words instead.
# ---------------------------------------------------------------------------
_CAD_VOCABULARY = {
    # Nouns (shapes)
    'cube': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'box': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'sphere': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'ellipsoid': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'table': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'object': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'square': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'rectangle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'triangle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'circle': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'cylinder': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'cone': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'tetrahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'hexahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'octahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'dodecahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'icosahedron': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'pyramid': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),
    'prism': vector_from_features("noun", loc=[0.0, 0.0, 0.0], scale=[0.0, 0.0, 0.0]),

    # Units
    'degree': vector_from_features("noun unit", number=1.0),  # angular unit
    'unit': vector_from_features("noun unit", number=1.0),   # generic unit
    'pixel': vector_from_features("noun unit", number=1.0),  # screen unit
    'meter': vector_from_features("noun unit", number=1.0),  # distance unit
    'inch': vector_from_features("noun unit", number=1.0),   # distance unit
    'foot': vector_from_features("noun unit", number=1.0),   # distance unit
    'yard': vector_from_features("noun unit", number=1.0),   # distance unit

    # Adjectives — colors
    'red': vector_from_features("adj", color=[1.0, 0.0, 0.0]),
    'green': vector_from_features("adj", color=[0.0, 1.0, 0.0]),
    'blue': vector_from_features("adj", color=[0.0, 0.0, 1.0]),
    'yellow': vector_from_features("adj", color=[1.0, 1.0, 0.0]),
    'purple': vector_from_features("adj", color=[0.5, 0.0, 0.5]),
    'orange': vector_from_features("adj", color=[1.0, 0.5, 0.0]),
    'black': vector_from_features("adj", color=[0.0, 0.0, 0.0]),
    'white': vector_from_features("adj", color=[1.0, 1.0, 1.0]),
    'gray': vector_from_features("adj", color=[0.5, 0.5, 0.5]),
    'brown': vector_from_features("adj", color=[0.6, 0.3, 0.1]),

    # Adjectives — sizes
    'large': vector_from_features("adj", scale=[2.0, 2.0, 2.0]),
    'big': vector_from_features("adj", scale=[2.0, 2.0, 2.0]),
    'huge': vector_from_features("adj", scale=[3.0, 3.0, 3.0]),
    'small': vector_from_features("adj", scale=[-0.5, -0.5, -0.5]),
    'tiny': vector_from_features("adj", scale=[-0.7, -0.7, -0.7]),
    'tall': vector_from_features("adj", scale=[0.0, 1.5, 0.0]),
    'short': vector_from_features("adj", scale=[0.0, -0.5, 0.0]),
    'wide': vector_from_features("adj", scale=[1.5, 0.0, 0.0]),
    'deep': vector_from_features("adj", scale=[0.0, 0.0, 1.5]),

    # Adjectives — textures / transparency
    'rough': vector_from_features("adj", texture=2.0),
    'smooth': vector_from_features("adj", texture=0.5),
    'shiny': vector_from_features("adj", texture=0.0),
    'clear': vector_from_features("adj", transparency=2.0),
    'transparent': vector_from_features("adj", transparency=2.0),
    'opaque': vector_from_features("adj", transparency=0.0),

    # Verbs — create
    'create': vector_from_features("verb action create"),
    'draw': vector_from_features("verb action create"),
    'make': vector_from_features("verb action create transform"),
    'build': vector_from_features("verb action create"),

    # Verbs — edit
    'copy': vector_from_features("verb action edit"),
    'delete': vector_from_features("verb action edit"),
    'remove': vector_from_features("verb action edit"),
    'paste': vector_from_features("verb action edit"),

    # Verbs — organize
    'place': vector_from_features("verb action organize"),
    'align': vector_from_features("verb action organize"),
    'group': vector_from_features("verb action organize"),
    'position': vector_from_features("verb action organize"),
    'ungroup': vector_from_features("verb action organize"),

    # Verbs — select
    'select': vector_from_features("verb action select"),

    # Verbs — style
    'color': vector_from_features("verb action transform style"),
    'texture': vector_from_features("verb action transform texture"),

    # Verbs — naming
    'call': vector_from_features("verb action naming"),
    'name': vector_from_features("verb action naming"),

    # Verbs — modify
    'move': vector_from_features("verb action transform move"),
    'rotate': vector_from_features("verb action transform rotate"),
    'xrotate': vector_from_features("verb action transform rotate", rotX=1.0),  # rotate around x-axis
    'yrotate': vector_from_features("verb action transform rotate", rotY=1.0),  # rotate around y-axis
    'zrotate': vector_from_features("verb action transform rotate", rotZ=1.0),  # rotate around z-axis
    'scale': vector_from_features("verb action transform scale"),

    # Verbs — generic (no third term)
    'redo': vector_from_features("verb action"),
    'undo': vector_from_features("verb action"),

    # Verbs / nouns — time travel
    'go': vector_from_features("verb action"),
    'time': vector_from_features("noun"),

    # Axis references
    'x': vector_from_features("noun", rotX=1.0),                                  # x-axis reference
    'y': vector_from_features("noun", rotY=1.0),                                  # y-axis reference
    'z': vector_from_features("noun", rotZ=1.0),                                  # z-axis reference
    'x-axis': vector_from_features("noun", rotX=1.0),                             # x-axis explicit
    'y-axis': vector_from_features("noun", rotY=1.0),                             # y-axis explicit
    'z-axis': vector_from_features("noun", rotZ=1.0),                             # z-axis explicit
}


# Engraf's full vocabulary = shared function words + CAD content.
SEMANTIC_VECTOR_SPACE = {**FUNCTION_WORDS, **_CAD_VOCABULARY}
