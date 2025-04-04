"""
Test: Validate SessionMetadata and EphysMetadata save/load via public helpers.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.path import get_rec_path
from utils.metadata import (
    load_session_metadata,
    load_ephys_metadata,
    SessionMetadata,
    EphysMetadata,
)
from utils.logger import logger

# --- Test parameters ---
rat_id = "NC40008"
session_name = "20250328_134136"
session_folder = get_rec_path(rat_id, session_name)

# --- Step 1: Load (or generate) SessionMetadata ---
session_1 = load_session_metadata(session_folder)
logger.log("Loaded SessionMetadata")

# --- Step 2: Load (or generate) EphysMetadata ---
ephys_save_path = session_1.extracted_dir / "ephys_metadata.pkl"
channel_map_path = session_1.rat_path / "ephys_channel_map_metadata.csv"

ephys_1 = load_ephys_metadata(
    rec_path=session_1.rec_path,
    channel_map_path=channel_map_path,
    save_path=ephys_save_path,
)
logger.log("Loaded EphysMetadata")

# --- Step 3: Reload from disk to verify persistence ---
session_2 = SessionMetadata.load(rat_id, session_name)
ephys_2 = EphysMetadata.load(ephys_save_path)

# --- Step 4: Compare fields ---
assert session_1.session_name == session_2.session_name
assert session_1.rec_path == session_2.rec_path
assert session_1.extracted_dir == session_2.extracted_dir
logger.log("✅ SessionMetadata consistency validated")

assert ephys_1.trodes_id == ephys_2.trodes_id
assert ephys_1.headstage_hardware_id == ephys_2.headstage_hardware_id
assert ephys_1.trodes_id_include == ephys_2.trodes_id_include
assert np.isclose(ephys_1.sampling_rate_hz, ephys_2.sampling_rate_hz)
logger.log("✅ EphysMetadata consistency validated")

# --- Done ---
logger.log("✅ All metadata save/load tests passed.")
