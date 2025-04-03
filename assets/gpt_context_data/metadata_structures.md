# Session Metadata Structure

## The `SessionMetadata` class

Creates a `session_metadata.pkl` file that contains:

- `session_name` (str): e.g., `20250328_134136`
- `animal_id` (str): e.g., `NC40008`
- `rec_path` (Path): path to the `.rec` file
- `trodesconf_path` (Path): path to the `.trodesconf` file
- `sampling_rate_hz` (float): extracted from config
- `exported_csc_channels` (list[int]): indices of all analog channels
- `export_paths` (dict): keys = "csc", "dio", "spikes", values = Path objects
- `extracted_dir` (Path): root path to processed data

We should have a way to add custom experiment specific data types adn fields to this through a method call in the class. 

This file is serialized as a `.pkl` and reloaded for all downstream tasks.

Should provide built in methods for saving and loading itself.

# Ephys Metadata Structure

## The `EphysMetadata` class

Stored as `ephys_metadata.pkl`, the contents of which are to be determined.

Should be initialized partly from `ephys_channel_map_metadata.csv`, which includes:

- analysis_index
- trodes_id
- headstage_hardware_id
- probe_hardware_id
- probe_id
- shank_id
- shank_site
- exclude

The `ephys_metadata.pkl` should include information related to:

- Indeces of Spike and CSC data, the actual data will stored only in the original .rec.
- Associate probe information (probe id, shank id, shank site) 
- Timestamp sycnronization info.

I know we will want to include timestamp syncronization info as:

- `timestamp_mapping` (dict): keys = "poly_coeffs", "r_squared", values = polynomial fit parameters mapping SpikeGadgets timestamps to ROS time.

This file is serialized as a `.pkl` and reloaded for all downstream tasks.