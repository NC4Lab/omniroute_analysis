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
load_dotenv(dotenv_path=env_path)

TRODES_PATH = os.getenv("TRODES_PATH")
NC4_DATA_PATH = os.getenv("NC4_DATA_PATH")

if not TRODES_PATH or not NC4_DATA_PATH:
    raise EnvironmentError("TRODES_PATH and NC4_DATA_PATH must be set in the .env file.")
