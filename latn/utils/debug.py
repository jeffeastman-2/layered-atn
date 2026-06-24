"""
Debug utilities for the ENGRAF system.
"""

# Global debug state
_debug_enabled = False


def set_debug(enabled: bool) -> None:
    """Enable or disable debug mode globally.
    
    Args:
        enabled: True to enable debug mode, False to disable
    """
    global _debug_enabled
    _debug_enabled = enabled


def is_debug() -> bool:
    """Check if debug mode is currently enabled.
    
    Returns:
        True if debug mode is enabled, False otherwise
    """
    return _debug_enabled


def debug_print(*args, **kwargs) -> None:
    """Print debug message only if debug mode is enabled.
    
    Args:
        *args: Arguments to pass to print()
        **kwargs: Keyword arguments to pass to print()
    """
    if _debug_enabled:
        print(*args, **kwargs)