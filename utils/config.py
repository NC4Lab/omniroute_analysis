"""
Module: config.py

Purpose:
    Load environment variables required across the project.
"""

from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env file explicitly from project root to ensure compatibility across modules
env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise FileNotFoundError(f".env file not found at: {env_path}")

TRODES_DIR = os.getenv("TRODES_DIR")
NC4_DATA_DIR = os.getenv("NC4_DATA_DIR")

if not TRODES_DIR or not NC4_DATA_DIR:
    raise EnvironmentError("TRODES_DIR and NC4_DATA_DIR must be set in the .env file.")
