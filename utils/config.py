from pathlib import Path
import os

class Session:
    """
    A lightweight object representing a single experimental session,
    with path utilities and metadata access.
    """

    def __init__(self, session_folder: str):
        """
        Initialize a session object from its root directory.

        Args:
            session_folder (str): Absolute or relative path to the session root.
        """
        self.session_path = Path(session_folder).resolve()

        # Parse metadata
        self.session_name = self.session_path.name
        self.animal_id = self._extract_animal_id()
        self.date = self._extract_date()

    def _extract_animal_id(self) -> str:
        """Extracts the animal ID (e.g., NC4012) from folder name or path."""
        for part in self.session_path.parts:
            if part.startswith("NC4"):
                return part
        raise ValueError("Animal ID (e.g., NC4xxxx) not found in session path.")

    def _extract_date(self) -> str:
        """Extracts the date string (YYYYMMDD) from folder name."""
        name = self.session_path.name
        try:
            return name.split("_")[0]  # e.g., '20250328'
        except Exception:
            raise ValueError(f"Failed to extract date from folder name: {name}")

    def get_extracted_folder(self, create: bool = False) -> Path:
        """Return the path to the extracted folder."""
        path = self.session_path / "extracted"
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_trodes_folder(self) -> Path:
        return self.session_path / "raw" / "Trodes"

    def get_ros_folder(self) -> Path:
        return self.session_path / "raw" / "ROS"

    def get_dio_path(self, channel: int) -> Path:
        """
        Return path to the DinXX.dat file extracted from Trodes.

        Args:
            channel (int): Digital input channel number (e.g., 1 for Din01)
        """
        dio_dir = self.get_extracted_folder() / f"{self.date}_merged.DIO"
        dio_file = dio_dir / f"Din{channel:02d}.dat"
        if not dio_file.exists():
            raise FileNotFoundError(f"DIO file not found: {dio_file}")
        return dio_file

    def get_ros_bag_path(self) -> Path:
        """
        Return the path to the main ROS bag file.
        Assumes one file exists in raw/ROS matching 'ExperimentData*.bag'.
        """
        bag_dir = self.get_ros_folder()
        bag_files = list(bag_dir.glob("ExperimentData*.bag"))
        if len(bag_files) == 0:
            raise FileNotFoundError("No ExperimentData bag file found in ROS folder.")
        elif len(bag_files) > 1:
            raise RuntimeError(f"Multiple ROS bag files found: {[str(f.name) for f in bag_files]}")
        return bag_files[0]

    def get_sync_fit_path(self, create: bool = True) -> Path:
        """
        Path where the polynomial sync fit will be saved/loaded from.
        """
        fit_path = self.get_extracted_folder(create=create) / "session_sync_fit.pkl"
        return fit_path

    def __repr__(self):
        return f"Session(path={self.session_path}, animal={self.animal_id}, date={self.date})"
