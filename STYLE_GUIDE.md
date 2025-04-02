# OMNIROUTE_ANALYSIS Code Style Guide

This guide outlines code commenting and structuring standards for collaborative development within the lab.

## 1. Function and Class Docstrings

All public functions and classes should have a docstring following this structure:

```
def example_function(param1: int, param2: str) -> bool:
    \"\"\"
    One-line summary of what the function does.

    Parameters:
        param1 (int): Description of first parameter.
        param2 (str): Description of second parameter.

    Returns:
        bool: Description of the return value.
    \"\"\"
```

## 2. Inline Comments

Use inline comments **only when the logic is non-obvious**, and focus on **why**, not just what:

```python
# Ensure the path exists before proceeding to load the data
if not rec_path.exists():
    raise FileNotFoundError(...)
```

## 3. Module-Level Docstrings

At the top of every `.py` file, include a brief summary (Note: don't include dependencies):

```
\"\"\"
Module: io_trodes.py

Purpose:
    Utilities for reading continuous data from SpikeGadgets .rec files using SpikeInterface.
\"\"\"
```

## 4. Naming Conventions

- Use `snake_case` for variables and function names
- Use `UPPER_CASE` for constants
- Use `CamelCase` for class names
