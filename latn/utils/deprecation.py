"""
Deprecation utilities for Engraf.

Provides decorators and functions to mark code as deprecated with appropriate warnings.
"""

import warnings
import functools


def deprecated(reason=None, replacement=None, version=None):
    """
    Decorator to mark functions/classes as deprecated.
    
    Args:
        reason: Optional reason for deprecation
        replacement: Optional suggested replacement function/method
        version: Optional version when deprecation started
    
    Usage:
        @deprecated("Use new_function() instead", replacement="new_function")
        def old_function():
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"{func.__name__} is deprecated"
            
            if version:
                message += f" (since version {version})"
                
            if reason:
                message += f". {reason}"
                
            if replacement:
                message += f" Use {replacement} instead."
                
            warnings.warn(
                message,
                DeprecationWarning,
                stacklevel=2
            )
            
            return func(*args, **kwargs)
        
        # Mark the wrapper as deprecated for inspection
        wrapper.__deprecated__ = True
        wrapper.__deprecation_info__ = {
            'reason': reason,
            'replacement': replacement,
            'version': version
        }
        
        return wrapper
    return decorator


def deprecated_module(reason=None, replacement=None, version=None):
    """
    Function to mark an entire module as deprecated.
    Call this at the top of the module file.
    
    Args:
        reason: Optional reason for deprecation
        replacement: Optional suggested replacement module
        version: Optional version when deprecation started
    """
    import inspect
    
    # Get the module name from the caller
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals.get('__name__', 'unknown')
    
    message = f"Module {module_name} is deprecated"
    
    if version:
        message += f" (since version {version})"
        
    if reason:
        message += f". {reason}"
        
    if replacement:
        message += f" Use {replacement} instead."
        
    warnings.warn(
        message,
        DeprecationWarning,
        stacklevel=2
    )


def is_deprecated(func):
    """
    Check if a function is marked as deprecated.
    
    Args:
        func: Function to check
        
    Returns:
        bool: True if function is deprecated
    """
    return hasattr(func, '__deprecated__') and func.__deprecated__


def get_deprecation_info(func):
    """
    Get deprecation information for a function.
    
    Args:
        func: Function to inspect
        
    Returns:
        dict or None: Deprecation info if available
    """
    if is_deprecated(func):
        return func.__deprecation_info__
    return None


# Context manager to temporarily suppress deprecation warnings
class suppress_deprecation_warnings:
    """
    Context manager to temporarily suppress deprecation warnings.
    
    Usage:
        with suppress_deprecation_warnings():
            old_function()  # Won't show deprecation warning
    """
    
    def __enter__(self):
        self.old_filters = warnings.filters[:]
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filters[:] = self.old_filters


__all__ = [
    'deprecated', 
    'deprecated_module', 
    'is_deprecated', 
    'get_deprecation_info',
    'suppress_deprecation_warnings'
]
