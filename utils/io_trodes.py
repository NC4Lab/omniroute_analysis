from utils.config import Session
from utils.io_binary import TrodesDIOBinaryLoader

def load_dio_events(session: Session, channel: int) -> list[float]:
    """
    Load rising edge timestamps from SpikeGadgets digital input (DIO) channel.

    Args:
        session (Session): Session object containing path and metadata
        channel (int): DIO channel number (e.g., 1 for Din01)

    Returns:
        List[float]: List of rising edge timestamps in seconds (SpikeGadgets timebase)
    """
    dio_path = session.get_dio_path(channel)
    dio_loader = TrodesDIOBinaryLoader(str(dio_path))
    rising_edges = dio_loader.dio['state'].astype(int).diff() == 1
    rising_ts = dio_loader.dio.index[rising_edges]
    return (rising_ts / float(dio_loader.clockrate)).tolist()
