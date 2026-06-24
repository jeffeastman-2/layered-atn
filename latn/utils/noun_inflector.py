import re

# Example plural-to-singular dictionary for irregulars
IRREGULAR_PLURALS = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "geese": "goose",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
}

def singularize_noun(word):
    """Convert plural noun to its singular form."""
    word_lower = word.lower()
    
    # Handle irregular plurals
    if word_lower in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word_lower], True
    
    # Handle -ves plurals (f/fe -> ves)
    if re.match(r".+ves$", word_lower):
        # Most -ves words come from -f or -fe words
        base = re.sub(r"ves$", "f", word_lower)
        # Check if it should be -fe (common case)
        if base in ['knif', 'lif', 'wif', 'calf', 'half', 'loaf']:
            base += 'e'
        return base, True
    
    # Handle -ies plurals (y -> ies)
    if re.match(r".+ies$", word_lower):
        return re.sub(r"ies$", "y", word_lower), True
    
    # Handle -es plurals (but not -ves or -ies which are handled above)
    # Only for words that actually need -es (ending in s, sh, ch, x, z, o)
    if (re.match(r".+[sxz]es$", word_lower) or 
        re.match(r".+[sc]hes$", word_lower) or 
        re.match(r".+oes$", word_lower)) and not word_lower.endswith("ss"):
        return re.sub(r"es$", "", word_lower), True
    
    # Handle regular -s plurals (but not -ss endings and not very short words)
    if (re.match(r".+s$", word_lower) and not word_lower.endswith("ss") and 
        len(word_lower) > 2):  # Don't treat short words like 'is', 'as' as plurals
        return re.sub(r"s$", "", word_lower), True
    
    # Word is not a plural
    return word, False

def is_plural(word):
    singular, is_plural_result = singularize_noun(word)
    return is_plural_result
