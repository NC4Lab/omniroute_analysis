"""
Module: path.py

Purpose:
    Utilities for constructing paths and/or resolving metadata.
"""

from pathlib import Path
from utils.config import NC4_DATA_PATH

def get_rec_path(rat_id: str, session_name: str) -> Path:
    """
    Build a standardized path to a `.rec` file given rat ID and session.

    Parameters:
        rat_id (str): Unique animal identifier (e.g., 'NC40008')
        session_name (str): Timestamped session name (e.g., '20250328_134136')

    Returns:
        Path: Full path to the expected `.rec` file
    """
    return (
        Path(NC4_DATA_PATH) /
        rat_id /
        session_name /
        "raw" /
        "Trodes" /
        f"{session_name}_merged.rec"
    )
