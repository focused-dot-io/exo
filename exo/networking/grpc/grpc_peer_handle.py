import grpc
import grpc.aio
import asyncio
from typing import Optional
import os

from exo.networking.peer_handle import PeerHandle
from exo.networking.grpc.node_service_pb2_grpc import NodeServiceStub
from exo.networking.grpc.node_service_pb2 import HealthCheckRequest

DEBUG = int(os.environ.get("DEBUG", "0"))
CONNECT_TIMEOUT = 30  # seconds

class GRPCPeerHandle(PeerHandle):
    """Handle for a GRPC peer connection."""

    def __init__(self, peer_id: str, addr: str, description: str, device_capabilities):
        super().__init__(peer_id, addr, description, device_capabilities)
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[NodeServiceStub] = None

    async def connect(self) -> None:
        """Connect to the peer."""
        if self.channel is not None:
            return

        try:
            # Create channel with timeout
            async with asyncio.timeout(CONNECT_TIMEOUT):
                self.channel = grpc.aio.insecure_channel(self.addr)
                await self.channel.channel_ready()
                self.stub = NodeServiceStub(self.channel)
        except Exception as e:
            if DEBUG >= 2:
                print(f"[GRPC Peer] Error connecting to {self.addr}: {e}")
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """Disconnect from the peer."""
        if self.channel is not None:
            try:
                await self.channel.close()
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[GRPC Peer] Error disconnecting from {self.addr}: {e}")
            finally:
                self.channel = None
                self.stub = None

    async def is_connected(self) -> bool:
        """Check if the peer is connected."""
        if self.channel is None:
            return False
        try:
            state = self.channel._channel.check_connectivity_state(True)
            return state == grpc.ChannelConnectivity.READY
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Check if the peer is healthy."""
        try:
            if not await self.is_connected():
                await self.connect()
            
            # Try a simple health check call
            await asyncio.wait_for(
                self.stub.HealthCheck(HealthCheckRequest()),
                timeout=CONNECT_TIMEOUT
            )
            return True
        except Exception as e:
            if DEBUG >= 2:
                print(f"[GRPC Peer] Health check failed for {self.peer_id}@{self.addr}: {e}")
            return False

    def __str__(self) -> str:
        return f"GRPCPeer({self.peer_id}@{self.addr})"

    def __repr__(self) -> str:
        return self.__str__()
