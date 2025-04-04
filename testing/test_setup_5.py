"""
Test: Setup and validate SessionMetadata, EphysMetadata, and CSC loading.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.metadata import SessionMetadata, EphysMetadata
from utils.io_trodes import load_csc_from_rec
from utils.logger import logger

# --- Test parameters ---
rat_id = "NC40008"
session_name = "20250328_134136"

# --- Step 1: Create and save SessionMetadata ---
session = SessionMetadata()
session.load_extract_data("NC40008", "20250328_134136")
logger.log("Initialized SessionMetadata")
session.save()
logger.log("SessionMetadata saved")

# --- Step 2: Reload and validate SessionMetadata ---
loaded_session = SessionMetadata()
loaded_session.load_extract_data(rat_id, session_name)
assert loaded_session.rat_id == rat_id
assert loaded_session.session_name == session_name
logger.log("SessionMetadata reload validated")

# --- Step 3: Create and save EphysMetadata ---
rec_path = loaded_session.rec_path
channel_map_path = loaded_session.rat_path / "ephys_channel_map_metadata.csv"
ephys_save_path = loaded_session.extracted_dir / "ephys_metadata.pkl"

ephys = EphysMetadata()
ephys.load_extract_data(
    rec_path=rec_path,
    channel_map_path=channel_map_path,
    save_path=ephys_save_path
)
logger.log(f"EphysMetadata initialized: {len(ephys.trodes_id)} channels available")
logger.log(f"Sampling rate: {ephys.sampling_rate_hz:.2f} Hz")

ephys.save()
logger.log("EphysMetadata saved")

# --- Step 4: Reload and validate EphysMetadata ---
reloaded_ephys = EphysMetadata()
reloaded_ephys.load_extract_data(
    rec_path=rec_path,
    channel_map_path=channel_map_path,
    save_path=ephys_save_path
)

assert reloaded_ephys.sampling_rate_hz == ephys.sampling_rate_hz
assert reloaded_ephys.trodes_id == ephys.trodes_id
assert reloaded_ephys.trodes_id_include == ephys.trodes_id_include
logger.log("EphysMetadata reload validated")

# --- Step 5: Load CSC data and assign directly ---
hardware_ids = ephys.trodes_to_headstage_ids(ephys.trodes_id_include)
ephys.raw_csc_data = load_csc_from_rec(
    rec_path=loaded_session.rec_path,
    trodes_id_include=hardware_ids,  # now correctly mapped and string-typed
)

# --- Step 6: Validate CSC structure ---
assert ephys.raw_csc_data is not None
num_channels = ephys.raw_csc_data.get_num_channels()
num_frames = ephys.raw_csc_data.get_num_frames()
assert num_channels == len(ephys.trodes_id_include)
logger.log(f"✅ Loaded CSC data: {num_channels} channels x {num_frames} samples")

# --- Done ---
logger.log("✅ All tests passed.")
