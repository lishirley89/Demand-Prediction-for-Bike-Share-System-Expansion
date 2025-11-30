#!/usr/bin/env python3
"""
Validate that all Python files in the code directory can be imported without errors.
This script is used both locally and in CI/CD pipelines.
"""

import sys
import importlib.util
from pathlib import Path

def validate_imports():
    """Validate imports for all Python files in the code directory."""
    code_dir = Path(__file__).parent.parent / 'code'
    
    if not code_dir.exists():
        print(f"Error: Code directory not found at {code_dir}")
        return False
    
    python_files = sorted(code_dir.glob('*.py'))
    
    if not python_files:
        print("No Python files found in code directory")
        return False
    
    errors = []
    successful = []
    
    for file_path in python_files:
        try:
            spec = importlib.util.spec_from_file_location('module', file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                successful.append(file_path.name)
                print(f'✓ {file_path.name}')
        except SyntaxError as e:
            error_msg = f'Syntax error: {e}'
            errors.append((file_path.name, error_msg))
            print(f'✗ {file_path.name}: {error_msg}')
        except ImportError as e:
            error_msg = f'Import error: {e}'
            errors.append((file_path.name, error_msg))
            print(f'✗ {file_path.name}: {error_msg}')
        except Exception as e:
            error_msg = f'Error: {e}'
            errors.append((file_path.name, error_msg))
            print(f'✗ {file_path.name}: {error_msg}')
    
    print(f'\nSummary:')
    print(f'  Successful: {len(successful)}/{len(python_files)}')
    print(f'  Errors: {len(errors)}/{len(python_files)}')
    
    if errors:
        print(f'\nFiles with errors:')
        for filename, error in errors:
            print(f'  - {filename}: {error}')
        return False
    
    return True

if __name__ == '__main__':
    success = validate_imports()
    sys.exit(0 if success else 1)

