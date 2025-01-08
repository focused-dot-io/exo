from pathlib import Path
import tempfile
import shutil
import asyncio
from datetime import timedelta
import os
from typing import Optional, Dict, List, Set, AsyncIterator

from exo.networking.grpc.node_service_pb2 import ShardChunk, TransferStatus
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.networking.peer_handle import PeerHandle
from exo.networking.coordinator import PeerCoordinator
from exo.networking.grpc.node_service_pb2 import Shard as ShardProto
from exo.download.repo_progress import RepoProgressEvent
from exo.download.event_handler import EventHandler
from exo.download.shard import Shard
from exo.download.exceptions import PeerConnectionError

DEBUG = int(os.environ.get("DEBUG", "0"))
CONNECT_TIMEOUT = 30  # seconds
TRANSFER_TIMEOUT = 300  # seconds

class P2PShardDownloader:
    """Downloads shards from peers using P2P."""

    def __init__(self, peers: List[PeerHandle], quick_check: bool = False):
        self.peers = peers
        self.quick_check = quick_check
        self._on_progress = EventHandler()
        self.failed_peers: Set[PeerHandle] = set()

    async def ensure_shard(self, shard: Shard) -> Path:
        """Ensure a shard is available locally, downloading it if necessary."""
        # Get peers that have this shard
        if DEBUG >= 2:
            print(f"[P2P Download] Searching for peers with shard {shard}")

        available_peers = []
        for peer in self.peers:
            try:
                if not await peer.is_connected():
                    await peer.connect()

                status = await peer.stub.GetShardStatus(shard.to_proto())
                if status.has_shard:
                    available_peers.append(peer)
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Error checking peer {peer}: {e}")
                continue

        if not available_peers:
            if DEBUG >= 2:
                print(f"[P2P Download] No peers found with shard {shard}")
            raise PeerConnectionError(f"No peers have shard {shard}")

        # Try each peer until one succeeds
        last_error = None
        for peer in available_peers:
            try:
                if DEBUG >= 2:
                    print(f"[P2P Download] Attempting download from peer {peer}")
                return await self._download_shard(peer, shard)
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Error downloading from peer {peer}: {e}")
                last_error = e
                continue

        raise PeerConnectionError(f"All peers failed to download shard {shard}. Last error: {last_error}")

    @property
    def on_progress(self) -> EventHandler:
        """Event handler for download progress updates."""
        return self._on_progress 