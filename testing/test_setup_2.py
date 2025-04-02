"""
Test Script: test_export_csc.py

Purpose:
    Test full pipeline for exporting and loading CSC data using the SessionData class.
"""

import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.metadata import SessionData
from utils.io_trodes import parse_trodesconf, export_csc_to_si_binary, load_csc_binary_recording
from utils.logger import logger

# --- SETUP ---
rat_id = "NC40008"
session_name = "20250328_134136"
session = SessionData(rat_id=rat_id, session_name=session_name)

# --- METADATA EXTRACTION ---
parse_trodesconf(session)
session.save()

# --- EXPORT ---
export_csc_to_si_binary(session)

# --- VERIFY ---
recording = load_csc_binary_recording(session)
logger.log(f"Channels: {recording.get_channel_ids()}")
logger.log(f"Duration (s): {recording.get_num_frames() / recording.get_sampling_frequency():.2f}")
logger.log(f"Sampling Rate (Hz): {recording.get_sampling_frequency()}")
