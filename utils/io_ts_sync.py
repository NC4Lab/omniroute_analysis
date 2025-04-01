import numpy as np
import pickle
from utils.config import Session

def align_timestamps(sg_ts: list[float], ros_ts: list[float]) -> np.ndarray:
    """
    Aligns SpikeGadgets and ROS timestamps using linear regression.

    Args:
        sg_ts (list of float): SpikeGadgets timestamps (DIO rising edges)
        ros_ts (list of float): ROS timestamps (digital sync topic)

    Returns:
        np.ndarray: Polynomial coefficients (for np.polyval)
    """
    if len(sg_ts) != len(ros_ts):
        raise ValueError(f"Mismatch in sync pulse count: {len(sg_ts)} SG vs {len(ros_ts)} ROS")

    fit = np.polyfit(sg_ts, ros_ts, deg=1)  # linear fit hardcoded
    return fit

def save_sync_fit(session: Session, fit: np.ndarray) -> None:
    """
    Save polynomial fit to extracted folder.

    Args:
        session (Session): Session object
        fit (np.ndarray): Polynomial coefficients
    """
    path = session.get_sync_fit_path(create=True)
    with open(path, "wb") as f:
        pickle.dump(fit, f)

def load_sync_fit(session: Session) -> np.ndarray:
    """
    Load saved polynomial fit from extracted folder.

    Args:
        session (Session): Session object

    Returns:
        np.ndarray: Polynomial coefficients
    """
    path = session.get_sync_fit_path()
    with open(path, "rb") as f:
        return pickle.load(f)
