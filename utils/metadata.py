"""
Module: metadata.py

Purpose:
    Metadata structures for handling session-level and electrophysiological context.
"""

import pickle
from typing import Any, Literal

import numpy as np
import pandas as pd
from pathlib import Path

from utils.path import get_rec_path, get_rat_path
from utils.io_trodes import load_sample_rate_from_rec


class SessionMetadata:
    """
    Holds session-level context and paths used across the pipeline.

    Parameters:
        rat_id (str): Unique identifier for the rat (e.g., "NC40008").
        session_name (str): Timestamped session name (e.g., "20250328_134136").
    """

    def __init__(self, rat_id: str, session_name: str):
        self.rat_id = rat_id
        self.session_name = session_name

        self.rat_path = get_rat_path(rat_id)
        self.rec_path = get_rec_path(rat_id, session_name)

        self.session_type: Literal["ephys", "behaviour"] = (
            "ephys" if self.rec_path.exists() else "behaviour"
        )

        self.extracted_dir = self.rat_path / session_name / "extracted"
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        self.custom_fields: dict[str, Any] = {}

    def set_custom_field(self, key: str, value: Any) -> None:
        self.custom_fields[key] = value

    def save(self) -> None:
        out_path = self.extracted_dir / "session_metadata.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(rat_id: str, session_name: str) -> "SessionMetadata":
        rat_path = get_rat_path(rat_id)
        extracted_dir = rat_path / session_name / "extracted"
        in_path = extracted_dir / "session_metadata.pkl"
        with open(in_path, "rb") as f:
            return pickle.load(f)


class EphysMetadata:
    """
    Holds electrophysiological channel metadata and maps available vs. active channels.

    Parameters:
        rec_path (Path): Path to the SpikeGadgets .rec file.
        channel_map_path (Path): Path to the channel metadata CSV.
        save_path (Path): Path to save the EphysMetadata pickle file.
    """

    def __init__(self, rec_path: Path, channel_map_path: Path, save_path: Path):
        self.rec_path = rec_path
        self.channel_map_path = channel_map_path
        self.save_path = save_path

        self.trodes_id: list[int] = []                 # Full list from CSV
        self.headstage_hardware_id: list[int] = []     # Parallel list to trodes_id
        self.trodes_id_include: list[int] = []         # Dynamic mask for analysis

        self.sampling_rate_hz = load_sample_rate_from_rec(self.rec_path)
        self.processed_csc_data: dict[str, Any] = {}
        self.timestamp_mapping: dict[str, Any] | None = None

        self._load_channel_map()

    def _load_channel_map(self) -> None:
        if not self.channel_map_path.exists():
            raise FileNotFoundError(f"Channel map not found: {self.channel_map_path}")

        df = pd.read_csv(self.channel_map_path)
        filtered = df[df["exclude"] == False]

        self.trodes_id = filtered["trodes_id"].tolist()
        self.headstage_hardware_id = filtered["headstage_hardware_id"].tolist()
        self.trodes_id_include = self.trodes_id.copy()

    def trodes_to_headstage_ids(self, ids: list[int]) -> list[str]:
        """
        Convert a list of trodes_id values to headstage_hardware_id strings.

        Parameters:
            ids (list[int]): Subset of trodes_id to convert.

        Returns:
            list[str]: Corresponding headstage_hardware_id values as strings.
        """
        mapping = {tid: str(hid) for tid, hid in zip(self.trodes_id, self.headstage_hardware_id)}
        try:
            return [mapping[tid] for tid in ids]
        except KeyError as e:
            raise ValueError(f"Invalid trodes_id: {e.args[0]} not found in known mapping.")

    def add_empty_processed_array(self, name: str, csc_dataset: np.ndarray) -> None:
        self.processed_csc_data[name] = csc_dataset

    def save(self) -> None:
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(pickle_path: Path) -> "EphysMetadata":
        with open(pickle_path, "rb") as f:
            return pickle.load(f)


