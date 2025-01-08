from abc import ABC, abstractmethod
from typing import Any

class PeerHandle(ABC):
    """Abstract base class for peer handles."""

    def __init__(self, peer_id: str, addr: str, description: str, device_capabilities: Any):
        self.peer_id = peer_id
        self.addr = addr
        self.description = description
        self.device_capabilities = device_capabilities

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the peer."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the peer."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if the peer is connected."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the peer is healthy."""
        pass
