from typing import List, Set
import os

from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.download.shard import Shard

DEBUG = int(os.environ.get("DEBUG", "0"))

class PeerCoordinator:
    """Coordinates peer discovery and connection management."""

    def __init__(self):
        self.peers: List[GRPCPeerHandle] = []
        self.failed_peers: Set[GRPCPeerHandle] = set()

    async def update_peers(self, new_peers: List[GRPCPeerHandle]) -> bool:
        """Update the list of available peers."""
        added = []
        removed = []
        updated = []
        unchanged = []
        to_disconnect = []
        to_connect = []

        # Find peers to add, update, or keep unchanged
        for peer in new_peers:
            if peer in self.peers:
                unchanged.append(peer)
            else:
                added.append(peer)
                to_connect.append(peer)

        # Find peers to remove
        for peer in self.peers:
            if peer not in new_peers:
                removed.append(peer)
                to_disconnect.append(peer)

        # Update internal state
        self.peers = new_peers.copy()
        self.failed_peers.clear()

        # Connect new peers
        for peer in to_connect:
            try:
                await peer.connect()
                updated.append(peer)
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[Coordinator] Error connecting to peer {peer}: {e}")
                self.failed_peers.add(peer)

        # Disconnect removed peers
        for peer in to_disconnect:
            try:
                await peer.disconnect()
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[Coordinator] Error disconnecting from peer {peer}: {e}")

        if DEBUG >= 2:
            print(f"update_peers: added={added} removed={removed} updated={updated} unchanged={unchanged} to_disconnect={to_disconnect} to_connect={to_connect}")

        did_peers_change = bool(added or removed or updated)
        if DEBUG >= 2:
            print(f"did_peers_change={did_peers_change}")

        return did_peers_change

    async def get_peers_with_shard(self, shard: Shard) -> List[GRPCPeerHandle]:
        """Get a list of peers that have the specified shard."""
        available_peers = []

        for peer in self.peers:
            if peer in self.failed_peers:
                continue

            try:
                if not await peer.is_connected():
                    await peer.connect()

                status = await peer.stub.GetShardStatus(shard.to_proto())
                if status.has_shard:
                    available_peers.append(peer)

            except Exception as e:
                if DEBUG >= 2:
                    print(f"[Coordinator] Error checking peer {peer} for shard {shard}: {e}")
                self.failed_peers.add(peer)

        return available_peers

    @property
    def connected_peers(self) -> List[GRPCPeerHandle]:
        """Get a list of currently connected peers."""
        return [p for p in self.peers if p not in self.failed_peers] 