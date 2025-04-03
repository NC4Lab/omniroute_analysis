"""
Module: metadata.py

Purpose:
    Defines the SessionMetadata class for managing per-session metadata and paths.
"""

import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
from utils.logger import logger
from utils.config import NC4_DATA_PATH


class SessionMetadata:
    """
    Represents a single data collection session.
    Manages all paths and metadata related to raw and processed files.
    """

    def __init__(self, rat_id: str, session_name: str):
        """
        Initialize a SessionMetadata object with resolved paths and empty metadata.

        Parameters:
            rat_id (str): Unique animal identifier.
            session_name (str): Timestamped session name.
        """
        self.rat_id = rat_id
        self.session_name = session_name
        self.timestamp_created = datetime.now().isoformat()

        # Core directories
        self.session_dir = Path(NC4_DATA_PATH) / self.rat_id / self.session_name
        self.extracted_dir = self.session_dir / "extracted"

        # Input files
        self.rec_path = self.session_dir / "raw" / "Trodes" / f"{session_name}_merged.rec"
        self.trodesconf_path = self.session_dir / "raw" / "Trodes" / f"{session_name}_merged.trodesconf"

        # Output metadata file
        self.session_pkl_path = self.extracted_dir / "session_metadata.pkl"

        # Standard export folders
        self.export_paths = {
            "csc": self.extracted_dir / f"{session_name}_merged.csc",
            "spikes": self.extracted_dir / f"{session_name}_merged.spikes",
            "dio": self.extracted_dir / f"{session_name}_merged.dio",
        }

        # Metadata fields
        self.exported_csc_channels: list[int] = []
        self.sampling_rate_hz: float | None = None

        # Ensure extracted folder exists immediately
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        logger.log(f"SessionMetadata initialized: {self.session_dir}")

    def save(self) -> None:
        """
        Save the current session object to the extracted/session_metadata.pkl.
        """
        with open(self.session_pkl_path, "wb") as f:
            pickle.dump(self, f)
        logger.log(f"SessionMetadata metadata saved to: {self.session_pkl_path}")

    @staticmethod
    def load(session_dir: Path) -> "SessionMetadata":
        """
        Load a session object from the session_dir/extracted/session_metadata.pkl file.

        Parameters:
            session_dir (Path): Full path to session directory.

        Returns:
            SessionMetadata: Loaded session object.
        """
        pkl_path = session_dir / "extracted" / "session_metadata.pkl"
        with open(pkl_path, "rb") as f:
            session = pickle.load(f)
        logger.log(f"SessionMetadata metadata loaded from: {pkl_path}")
        return session

    @staticmethod
    def from_dir(session_dir: Path) -> "SessionMetadata":
        """
        Construct a SessionMetadata object from a full path like .../<rat_id>/<session_name>

        Parameters:
            session_dir (Path): Full session directory path.

        Returns:
            SessionMetadata: A new SessionMetadata instance.
        """
        rat_id = session_dir.parts[-2]
        session_name = session_dir.parts[-1]
        return SessionMetadata(rat_id, session_name)

    def to_dict(self) -> dict:
        """
        Return a dictionary representation of key session metadata.

        Returns:
            dict: SessionMetadata details including paths and core fields.
        """
        return {
            "rat_id": self.rat_id,
            "session_name": self.session_name,
            "timestamp_created": self.timestamp_created,
            "session_dir": str(self.session_dir),
            "extracted_dir": str(self.extracted_dir),
            "rec_path": str(self.rec_path),
            "trodesconf_path": str(self.trodesconf_path),
            "session_pkl_path": str(self.session_pkl_path),
            "exported_csc_channels": self.exported_csc_channels,
            "sampling_rate_hz": self.sampling_rate_hz,
            "export_paths": {k: str(v) for k, v in self.export_paths.items()},
        }

class EphysMetadata:
    """
    Loads and filters channel metadata from a session-specific CSV.
    """

    def __init__(self, session, include_only: bool = True):
        """
        Initialize and optionally filter to only included channels.

        Parameters:
            session (SessionMetadata): A loaded SessionMetadata object.
            include_only (bool): If True, apply 'exclude == False' filter by default.
        """
        self.csv_path = session.extracted_dir / "channel_data.csv"

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Channel metadata CSV not found at {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if include_only:
            self.df = self.df[self.df["exclude"] == False]

    def filter(self, **criteria) -> list[int]:
        """
        Return list of channel_ids matching all specified filters.

        Example:
            filter(region="CA1", quality="good")
        """
        df_filtered = self.df.copy()
        for key, value in criteria.items():
            if key not in df_filtered.columns:
                raise ValueError(f"Column '{key}' not found in channel metadata.")
            df_filtered = df_filtered[df_filtered[key] == value]
        return df_filtered["channel_id"].tolist()