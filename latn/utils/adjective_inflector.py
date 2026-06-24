"""
Adjective inflection utilities for ENGRAF.

Handles comparative and superlative forms of adjectives:
- Comparative: bigger, redder, taller, etc.
- Superlative: biggest, reddest, tallest, etc.

This module provides functions to detect inflected adjectives and convert them
back to their base forms with appropriate morphological marking.
"""

from typing import Tuple


def base_adjective_from_comparative(word: str) -> Tuple[str, str]:
    """Convert comparative/superlative adjective to base form.
    
    Args:
        word: The adjective to analyze
        
    Returns:
        Tuple of (base_form, form_type) where form_type can be:
        - 'base': not a comparative/superlative form
        - 'comparative': -er form (bigger, taller, etc.)
        - 'superlative': -est form (biggest, tallest, etc.)
    """
    word = word.lower()
    
    # Handle irregular comparative/superlative forms first
    irregular_forms = {
        "better": ("good", "comparative"),
        "best": ("good", "superlative"),
        "worse": ("bad", "comparative"),
        "worst": ("bad", "superlative"),
        "more": ("much", "comparative"),
        "most": ("much", "superlative"),
        "further": ("far", "comparative"),
        "furthest": ("far", "superlative"),
        "farther": ("far", "comparative"),
        "farthest": ("far", "superlative")
    }
    
    if word in irregular_forms:
        return irregular_forms[word]
    
    # Handle -er endings (comparative forms)
    if word.endswith('er'):
        # Special cases where we need to handle doubled consonants
        if word.endswith('gger'):  # bigger -> big
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('tter'):  # fatter -> fat
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('nner'):  # thinner -> thin
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('dder'):  # redder -> red
            base = word[:-3]
            return base, 'comparative'
        elif word.endswith('er'):
            base = word[:-2]
            return base, 'comparative'
    
    # Handle -est endings (superlative forms)
    if word.endswith('est'):
        # Special cases where we need to handle doubled consonants
        if word.endswith('ggest'):  # biggest -> big
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('ttest'):  # fattest -> fat
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('nnest'):  # thinnest -> thin
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('ddest'):  # reddest -> red
            base = word[:-4]
            return base, 'superlative'
        elif word.endswith('est'):
            base = word[:-3]
            return base, 'superlative'
    
    # Not a comparative/superlative
    return word, 'base'


def is_comparative(word: str) -> bool:
    """Check if a word is a comparative adjective.
    
    Args:
        word: The word to check
        
    Returns:
        True if the word is a comparative form
    """
    _, form_type = base_adjective_from_comparative(word)
    return form_type == 'comparative'


def is_superlative(word: str) -> bool:
    """Check if a word is a superlative adjective.
    
    Args:
        word: The word to check
        
    Returns:
        True if the word is a superlative form
    """
    _, form_type = base_adjective_from_comparative(word)
    return form_type == 'superlative'


def get_base_adjective(word: str) -> str:
    """Get the base form of a potentially inflected adjective.
    
    Args:
        word: The adjective to get the base form of
        
    Returns:
        The base form of the adjective
    """
    base, _ = base_adjective_from_comparative(word)
    return base


def get_adjective_form_type(word: str) -> str:
    """Get the morphological form type of an adjective.
    
    Args:
        word: The adjective to analyze
        
    Returns:
        The form type: 'base', 'comparative', or 'superlative'
    """
    _, form_type = base_adjective_from_comparative(word)
    return form_type


def is_adjective_inflection(word: str) -> bool:
    """Check if a word is an inflected adjective (comparative or superlative).
    
    Args:
        word: The word to check
        
    Returns:
        True if the word is a comparative or superlative form
    """
    form_type = get_adjective_form_type(word)
    return form_type in ['comparative', 'superlative']


def find_root_adjective(word: str) -> Tuple[str, str, bool]:
    """Find the root adjective for a potentially inflected form.
    
    This function provides a consistent interface similar to find_root_verb.
    
    Args:
        word: The word to analyze
        
    Returns:
        Tuple of (root_adjective, inflection_type, found_root) where:
        - root_adjective: The base form of the adjective
        - inflection_type: 'comp' for comparative, 'super' for superlative, None for base
        - found_root: True if inflection was detected
    """
    base, form_type = base_adjective_from_comparative(word)
    
    if form_type == 'comparative':
        return base, 'comp', True
    elif form_type == 'superlative':
        return base, 'super', True
    else:
        return word, None, False


__all__ = [
    'base_adjective_from_comparative',
    'is_comparative',
    'is_superlative', 
    'get_base_adjective',
    'get_adjective_form_type',
    'is_adjective_inflection',
    'find_root_adjective'
]
