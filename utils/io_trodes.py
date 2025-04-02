"""
Module: io_trodes.py

Purpose:
    Handle SpikeGadgets .rec file export and loading using SpikeInterface and session metadata.
"""

from spikeinterface.extractors import read_spikegadgets, BinaryRecordingExtractor
from spikeinterface import write_binary_recording
from pathlib import Path
import xml.etree.ElementTree as ET

from utils.metadata import SessionData
from utils.logger import logger


def parse_trodesconf(session: SessionData) -> None:
    """
    Populate the session object with analog CSC channel indices based on the .trodesconf file.

    Parameters:
        session (SessionData): The session to update.
    """
    from utils.logger import logger

    with logger.time_block("Parsing Trodesconf"):
        tree = ET.parse(session.trodesconf_path)
        root = tree.getroot()

        csc_channels = []
        all_channels = root.findall(".//Channel")

        for i, ch in enumerate(all_channels):
            if ch.attrib.get("dataType", "").lower() == "analog":
                csc_channels.append(i)

        session.exported_csc_channels = csc_channels

        # Attempt to get sampling rate
        srate_node = root.find(".//HardwareConfiguration/SampleRate")
        if srate_node is not None and srate_node.text:
            session.sampling_rate_hz = float(srate_node.text)

def export_csc_to_si_binary(session: SessionData) -> None:
    """
    Export selected CSC channels from a .rec file to SI-compatible binary format.

    Parameters:
        session (SessionData): Active session object with populated metadata.
    """
    with logger.time_block("Exporting CSC to binary"):
        output_dir = session.export_paths["csc"]

        logger.log("Reading .rec file using SpikeInterface")
        recording = read_spikegadgets(session.rec_path)

        csc_channels = session.exported_csc_channels
        if not csc_channels:
            raise ValueError("No CSC channels defined in session.")

        logger.log(f"Filtering for {len(csc_channels)} CSC channels")
        filtered = recording.channel_slice(channel_ids=csc_channels)

        logger.log(f"Saving binary to: {output_dir}")
        write_binary_recording(
            recording=filtered,
            save_path=output_dir / "csc_interleaved.dat",
            dtype="int16",
            time_axis=1,
            overwrite=True
        )


def load_csc_binary_recording(session: SessionData) -> BinaryRecordingExtractor:
    """
    Load a previously exported CSC binary recording using session metadata.

    Parameters:
        session (SessionData): Loaded session object.

    Returns:
        BinaryRecordingExtractor: SpikeInterface-compatible object.
    """
    dat_path = session.export_paths["csc"] / "csc_interleaved.dat"
    logger.log(f"Loading CSC binary recording from: {dat_path}")
    return BinaryRecordingExtractor(
        dat_path,
        sampling_frequency=session.sampling_rate_hz,
        num_channels=len(session.exported_csc_channels),
        dtype="int16",
        time_axis=1
    )
