import sys
import time
from pathlib import Path
import time

# Add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.io_trodes import load_continuous_from_rec
from utils.path import get_rec_path

rat_id = "NC40008"
session_name = "20250328_134136"

# Time the full process
start_time = time.time()

# Resolve full .rec file path
rec_file = get_rec_path(rat_id, session_name)
print(f"Resolved path in {time.time() - start_time:.2f}s")

# Load the recording object (lazy-loaded)
load_start = time.time()
recording = load_continuous_from_rec(rec_file)
print(f"Loaded RecordingExtractor in {time.time() - load_start:.2f}s")

# Print metadata
print("Channels:", recording.get_channel_ids())
print("Duration (s):", recording.get_num_frames() / recording.get_sampling_frequency())
print("Sampling Rate (Hz):", recording.get_sampling_frequency())

print(f"Total time: {time.time() - start_time:.2f}s")

# Sampling rate (Hz)
fs = recording.get_sampling_frequency()

# Number of samples in 1 second
n_samples = int(fs)

# Select the first 10 channel IDs
channel_ids = recording.get_channel_ids()[:10]

# Time the data access
start = time.time()
snippet = recording.get_traces(
    channel_ids=channel_ids,
    start_frame=0,
    end_frame=n_samples
)
elapsed = time.time() - start

print(f"Loaded 1s of data from 10 channels in {elapsed:.4f} seconds. Shape: {snippet.shape}")
