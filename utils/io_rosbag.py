from pathlib import Path
from typing import List
from utils.config import Session
from datetime import datetime

from rosbags.highlevel import AnyReader as RosBagReader

def get_ros_bag_start_date(session: Session) -> str:
    """
    Extract the start date of the ROS bag in YYMMDD format.

    Args:
        session (Session): Session object with access to ROS bag

    Returns:
        str: Date string in YYMMDD format (e.g., '240329')
    """
    bag_path = Path(session.get_ros_bag_path())

    with RosBagReader([bag_path]) as reader:
        reader.open()
        # Get UNIX start time from bag metadata
        start_time_unix = reader.start_time  # float: UNIX timestamp
        dt = datetime.utcfromtimestamp(start_time_unix)
        return dt.strftime("%y%m%d")

def load_ros_sync_events(session: Session, topic: str = '/sync') -> List[float]:
    """
    Load timestamps of sync pulses from a ROS bag file using rosbags.highlevel (AnyReader).

    Args:
        session (Session): Session object containing ROS path
        topic (str): Topic name to extract sync events from (default: '/sync')

    Returns:
        List[float]: Timestamps of sync events in ROS timebase (seconds)
    """
    bag_path = Path(session.get_ros_bag_path())
    timestamps_ros: List[float] = []

    with RosBagReader([bag_path]) as reader:
        reader.open()

        if topic not in reader.topics:
            raise ValueError(f"Topic '{topic}' not found in ROS bag: {bag_path}")

        # Iterate through messages on the topic
        for conn, timestamp, rawdata in reader.messages(connections=reader.connections_by_topic[topic]):
            msg = reader.deserialize(rawdata, conn.msgtype)

            # Handle time from msg or msg.data
            if hasattr(msg, 'data'):
                sec = getattr(msg.data, 'sec', 0)
                nsec = getattr(msg.data, 'nanosec', 0)
            else:
                sec = getattr(msg, 'sec', 0)
                nsec = getattr(msg, 'nanosec', 0)

            timestamps_ros.append(float(sec) + float(nsec) * 1e-9)

    return timestamps_ros
