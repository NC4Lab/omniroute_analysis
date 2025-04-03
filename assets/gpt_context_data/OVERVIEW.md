# Overview of project

You are helping me develop a neuroscience data analysis pipeline for working with SpikeGadgets .rec files and ROS .bag files. The goal is to analyze both spike data and continuous LFP-like data (CSC — Continuous Sampled Channels), and to synchronize all data streams into a common timebase using digital sync pulses. This code base will likely be extended to include behavioral data. The final goal is a modular, efficient, and reproducible pipeline for large-scale neural data processing in Python, designed around SpikeInterface (v0.102.1) and tailored to our lab’s fixed environment and workflows.

## Ephys data

We are using SpikeGadgets wireless headstages and silicon probes, not tetrodes. The main data types of interest are:

- CSC (LFP-like) data from Trodes .rec files.
- Spikes (via SpikeInterface sorting modules).
- DIO sync events from both `.rec` and `.bag` files.
- Timebase mappings from SpikeGadgets time to ROS time.

## Pipeline and workflow

The pipeline is built with Python in VS Code and emphasizes:

- Modularity (each function and utility is its own clear unit).
- Compatibility with SpikeInterface (Kilosort clustering, waveform extraction, filtering, curation, etc.).
- Utilize the SpikeGadgets .rec files for all tasks and avoid unnecessary storage of derived data — everything can be re-computed.
- A consistent folder and naming convention for all session exports.
- Lightweight logging (utils/logger.py)  with consistent time tracking and structure visualization.
- Environment-specific path configuration via a `.env` file parsed by config.py.
- Use of metadata classes (e.g., SessionMetadata, EphysMetadata) to manage paths, configuration, and shared state across modules.

## Timestamp Synchronization

All data will be aligned to the ROS `.bag` timebase using digital sync pulses embedded in both data streams. This involves:

- Extracting sync pulse onsets from both `.rec` and `.bag` files
- Fitting a polynomial regression model to align SpikeGadgets time to ROS time
- Applying this mapping to spikes, CSC, and any behavioral or event data


## Need to know

What you need to know before proceeding:

- This codebase is not intended as a general-purpose framework — it is built around fixed expectations and conventions for our lab.
- We value simplicity and hardcoded structure over generality.
- Your job is to help write clean, modular functions that respect our design choices.
- Minimize unnecessary parameters or flexibility — assume `.env`, config paths, and session layout are consistent and stable.
- I will be sharing code as needed from an existing repository that has some of the functionality we want.
- The `.env` contains base paths and device-specific configuration, parsed by config.py. All code assumes this has been set correctly and does not require dynamic path resolution.
- Refer to the working documents `codebase_folder_structure.txt` and `session_level_folder_structure.txt` for directory layout and file organization.
- Refer to the working document `metadata_structures.md` for metadata classes and data files.
- Refer to the working document `STYLE_GUIDE.md` for code formatting.

## Useful links:
- SpikeInterface repo: https://github.com/SpikeInterface/spikeinterface?utm_source=chatgpt.com