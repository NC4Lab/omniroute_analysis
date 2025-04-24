"""
Script: export_csc_to_matlab.py

Purpose:
    Export CSC traces and aligned timestamps to a MATLAB .mat file
    for downstream analysis in MATLAB. Hardcoded for a specific session.
"""

from pathlib import Path
import numpy as np
from scipy.io import savemat
from utils.metadata import SessionMetadata, EphysMetadata
from utils.io_trodes import load_csc_from_rec, get_csc_sg_ts
from utils.ts_sync import convert_sg_ts_to_ros_time
from utils.omni_anal_logger import omni_anal_logger

# ----------------------------
# Hardcoded session + export path
# ----------------------------
rat_id = "NC40008"
session_name = "20250328_134136"
export_dir = Path(
    r"C:\Users\lester\UBC\Madhav, Manu - lesterkaur2024gate\analysis\gate_ephys_test\data\NC40008\20250328_134136\processed\matlab_csc"
)
overwrite = True
include_ros_ts = True

# ----------------------------
# Load metadata and CSC recording
# ----------------------------
omni_anal_logger.info("========== EXPORT CSC TO MATLAB ==========")
omni_anal_logger.info(f"Starting export for session: {rat_id}/{session_name}")
omni_anal_logger.info("Loading SessionMetadata and EphysMetadata...")

session_meta = SessionMetadata(rat_id, session_name)
session_meta.load_or_initialize_pickle()

ephys_meta = EphysMetadata(rat_id, session_name)
ephys_meta.load_or_initialize_pickle()

sampling_rate = ephys_meta.sampling_rate_hz
trodes_ids = ephys_meta.channel_trodes_id

omni_anal_logger.info(f"Sampling rate: {sampling_rate} Hz")
omni_anal_logger.info(f"Channel count: {len(trodes_ids)}")

omni_anal_logger.info("Loading CSC traces from .rec file using channel mapping...")
recording = load_csc_from_rec(
    rec_path=session_meta.rec_path,
    channel_trodes_id=ephys_meta.channel_trodes_id,
    channel_headstage_hardware_id=ephys_meta.channel_headstage_hardware_id,
    trodes_id_include=None
)

traces = recording.get_traces()
omni_anal_logger.info(f"Loaded CSC traces: shape = {traces.shape}")

# ----------------------------
# Compute SpikeGadgets timestamps
# ----------------------------
omni_anal_logger.info("Computing SpikeGadgets timebase timestamps...")
trodes_ts = get_csc_sg_ts(n_samples=traces.shape[0], sampling_rate_hz=sampling_rate)
omni_anal_logger.info(f"Generated {len(trodes_ts)} timestamps (SG timebase)")

# ----------------------------
# Optionally compute ROS-aligned timestamps
# ----------------------------
ros_ts = None
if include_ros_ts:
    omni_anal_logger.info("Attempting to compute ROS-aligned timestamps...")
    if not hasattr(ephys_meta, "timestamp_mapping") or ephys_meta.timestamp_mapping is None:
        omni_anal_logger.warning("No timestamp mapping found in ephys_meta — skipping ROS timestamps.")
    else:
        ros_ts = convert_sg_ts_to_ros_time(trodes_ts, sync_mapping=ephys_meta.timestamp_mapping)
        omni_anal_logger.info(f"Computed {len(ros_ts)} timestamps in ROS timebase")

# ----------------------------
# Prepare MATLAB-compatible export structure
# ----------------------------
omni_anal_logger.info("Preparing export dictionary for MATLAB .mat file...")
mat_data = {
    "csc_data": traces.astype(np.float32),       # (n_samples x n_channels)
    "trodes_ts": trodes_ts.astype(np.float64),   # (n_samples)
    "trodes_ids": np.array(trodes_ids),          # (n_channels)
    "sampling_rate": sampling_rate               # scalar
}
if ros_ts is not None:
    mat_data["ros_ts"] = ros_ts.astype(np.float64)

# ----------------------------
# Save to .mat file
# ----------------------------
export_dir.mkdir(parents=True, exist_ok=True)
export_path = export_dir / f"{session_name}_csc_export.mat"

if export_path.exists() and not overwrite:
    omni_anal_logger.warning(f"MAT file already exists at {export_path}. Skipping save.")
else:
    omni_anal_logger.info(f"Saving .mat file to: {export_path}")
    savemat(export_path, mat_data)
    omni_anal_logger.info("Export complete.")

omni_anal_logger.info("========== EXPORT FINISHED ==========")
