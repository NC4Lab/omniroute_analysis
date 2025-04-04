"""
Module: io_trodes.py

Purpose:
    Utilities for reading continuous data from SpikeGadgets .rec files using SpikeInterface.
"""

from pathlib import Path
import spikeinterface.extractors as se
from utils.logger import logger


def load_sample_rate_from_rec(rec_path: Path) -> None:
    """
    Load sampling rate from the .rec file and store it in the EphysMetadata.

    Parameters:
        rec_path (Path): Path to the .rec file.
    """
    rec = se.read_spikegadgets(rec_path)
    return rec.get_sampling_frequency()


def load_csc_from_rec(rec_path: Path, trodes_id_include: list[int]) -> None:
    """
    Load CSC data from a .rec file and store it in the EphysMetadata object.

    Parameters:
        rec_path (Path): Path to the .rec file.
        trodes_id_include (list[int]): List of Trodes channel IDs to include.
        ephys (EphysMetadata): Metadata object to store CSC data in.
    """
    with logger.time_block("Loading CSC data from .rec file"):
        rec = se.read_spikegadgets(rec_path)
        return rec.channel_slice(channel_ids=trodes_id_include)
