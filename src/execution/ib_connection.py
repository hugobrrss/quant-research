"""
Interactive Brokers connection management.
"""

import logging
from ib_insync import *

logger = logging.getLogger(__name__)


class IBConnection:
    """
    Manages connection to Interactive Brokers TWS/Gateway.

    Usage:
        conn = IBConnection()
        conn.connect()
        # ... do stuff ...
        conn.disconnect()

    Or as context manager:
        with IBConnection() as ib:
            # ... do stuff with ib ...
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,  # 7497 = paper, 7496 = live
        client_id: int = 1
    ):
        """
        Args:
            host: TWS host (localhost for local TWS)
            port: TWS port (7497 for paper trading, 7496 for live)
            client_id: Unique client identifier (increment if running multiple connections)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    def connect(self) -> IB:
        """Establish connection to TWS."""
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            logger.info(f"Connected to IB on {self.host}:{self.port}")
            logger.info(f"Account: {self.ib.managedAccounts()}")
            return self.ib
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            raise

    def disconnect(self):
        """Disconnect from TWS."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.ib.isConnected()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self.ib

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
