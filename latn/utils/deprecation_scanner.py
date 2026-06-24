#!/usr/bin/env python3
"""
Deprecation scanner for Engraf project.

Scans the codebase for usage of deprecated functions and provides a report.
"""

import ast
import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

from latn.utils.deprecation import is_deprecated, get_deprecation_info


class DeprecationVisitor(ast.NodeVisitor):
    """AST visitor to find function calls and imports."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.usages = []
        
    def visit_Call(self, node):
        """Visit function calls."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self.usages.append({
                'type': 'call',
                'name': func_name,
                'line': node.lineno,
                'col': node.col_offset
            })
        elif isinstance(node.func, ast.Attribute):
            attr_name = node.func.attr
            self.usages.append({
                'type': 'method_call',
                'name': attr_name,
                'line': node.lineno,
                'col': node.col_offset
            })
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Visit 'from module import ...' statements."""
        if node.names:
            for alias in node.names:
                name = alias.name
                self.usages.append({
                    'type': 'import',
                    'name': name,
                    'module': node.module,
                    'line': node.lineno,
                    'col': node.col_offset
                })
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Visit 'import ...' statements."""
        for alias in node.names:
            self.usages.append({
                'type': 'import_module',
                'name': alias.name,
                'line': node.lineno,
                'col': node.col_offset
            })
        self.generic_visit(node)


def scan_file(file_path: Path) -> List[Dict]:
    """Scan a single Python file for deprecated function usage."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=str(file_path))
        visitor = DeprecationVisitor(str(file_path))
        visitor.visit(tree)
        
        return visitor.usages
        
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return []


def scan_directory(directory: Path, exclude_patterns: List[str] = None) -> Dict[str, List[Dict]]:
    """Scan all Python files in a directory for deprecated function usage."""
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '.git', '.pytest_cache', 'build', 'dist']
    
    results = {}
    
    for file_path in directory.rglob('*.py'):
        # Skip excluded directories
        if any(pattern in str(file_path) for pattern in exclude_patterns):
            continue
            
        usages = scan_file(file_path)
        if usages:
            results[str(file_path)] = usages
            
    return results


def check_deprecated_functions() -> Dict[str, Dict]:
    """Check known deprecated functions in the Engraf codebase."""
    deprecated_funcs = {}
    
    # Check tokenize function
    try:
        from latn.lexer.token_stream import tokenize
        if is_deprecated(tokenize):
            info = get_deprecation_info(tokenize)
            deprecated_funcs['tokenize'] = {
                'module': 'latn.lexer.token_stream',
                'info': info,
                'function': tokenize
            }
    except ImportError:
        pass
        
    # Add other deprecated functions here as needed
    
    return deprecated_funcs


def generate_report():
    """Generate a deprecation usage report."""
    print("=" * 60)
    print("ENGRAF DEPRECATION SCANNER REPORT")
    print("=" * 60)
    print()
    
    # Check for deprecated functions
    deprecated_funcs = check_deprecated_functions()
    
    if deprecated_funcs:
        print("DEPRECATED FUNCTIONS DETECTED:")
        print("-" * 40)
        for func_name, info in deprecated_funcs.items():
            print(f"Function: {func_name}")
            print(f"Module: {info['module']}")
            dep_info = info['info']
            if dep_info:
                if dep_info.get('reason'):
                    print(f"Reason: {dep_info['reason']}")
                if dep_info.get('replacement'):
                    print(f"Replacement: {dep_info['replacement']}")
                if dep_info.get('version'):
                    print(f"Since version: {dep_info['version']}")
            print()
    
    # Scan for usage
    project_root = Path(__file__).parent.parent
    print(f"Scanning directory: {project_root}")
    print()
    
    results = scan_directory(project_root)
    
    # Filter for known deprecated functions
    deprecated_names = set(deprecated_funcs.keys())
    
    print("DEPRECATED FUNCTION USAGE:")
    print("-" * 40)
    
    total_usages = 0
    for file_path, usages in results.items():
        file_usages = []
        for usage in usages:
            if usage['name'] in deprecated_names:
                file_usages.append(usage)
                total_usages += 1
        
        if file_usages:
            print(f"\nFile: {file_path}")
            for usage in file_usages:
                print(f"  Line {usage['line']:3d}: {usage['type']} - {usage['name']}")
    
    if total_usages == 0:
        print("No usages of deprecated functions found!")
    else:
        print(f"\nTotal deprecated function usages: {total_usages}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    generate_report()
