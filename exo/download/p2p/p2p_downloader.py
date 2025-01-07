import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.networking.peer_handle import PeerHandle
from exo.networking.grpc.file_service_pb2 import (
    GetShardStatusRequest, GetShardStatusResponse,
    ShardChunk, TransferStatus
)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

class P2PShardDownloader(ShardDownloader):
    def __init__(self, peers: List[PeerHandle], quick_check: bool = False):
        self.peers = peers
        self.quick_check = quick_check
        self.active_downloads: Dict[Shard, asyncio.Task] = {}
        self.completed_downloads: Dict[Shard, Path] = {}
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()
        self.current_shard: Optional[Shard] = None

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        self.current_shard = shard
        
        if shard in self.completed_downloads:
            if DEBUG >= 2:
                print(f"Shard {shard} already downloaded at {self.completed_downloads[shard]}")
            return self.completed_downloads[shard]

        # Check if download already in progress
        if shard in self.active_downloads:
            if DEBUG >= 2:
                print(f"Download already in progress for shard {shard}")
            return await self.active_downloads[shard]

        # Find peers that have this shard
        available_peers = []
        if DEBUG >= 2:
            print(f"Searching for peers with shard {shard}")
            
        for peer in self.peers:
            try:
                if DEBUG >= 2:
                    print(f"Checking peer {peer} for shard {shard}")
                status = await peer.file_service.GetShardStatus(
                    GetShardStatusRequest(
                        shard=shard.to_proto(),
                        inference_engine_name=inference_engine_name
                    )
                )
                if status.has_shard:
                    if DEBUG >= 2:
                        print(f"Peer {peer} has shard {shard} at {status.local_path} (size: {status.file_size})")
                    available_peers.append((peer, status))
            except Exception as e:
                if DEBUG >= 2:
                    print(f"Error checking shard status on peer {peer}: {e}")
                continue

        if not available_peers:
            if DEBUG >= 2:
                print(f"No peers found with shard {shard}")
            raise FileNotFoundError(f"No peers have shard {shard}")

        # Choose peer with shard (for now just take first available)
        chosen_peer, status = available_peers[0]
        if DEBUG >= 2:
            print(f"Selected peer {chosen_peer} to download shard {shard}")
        
        # Start download
        download_task = asyncio.create_task(
            self._download_shard(shard, inference_engine_name, chosen_peer)
        )
        self.active_downloads[shard] = download_task
        
        try:
            path = await download_task
            self.completed_downloads[shard] = path
            if DEBUG >= 2:
                print(f"Successfully downloaded shard {shard} to {path}")
            return path
        finally:
            if shard in self.active_downloads:
                self.active_downloads.pop(shard)

    async def _download_shard(
        self, shard: Shard, inference_engine_name: str, peer: PeerHandle
    ) -> Path:
        if DEBUG >= 2:
            print(f"Starting download of shard {shard} from peer {peer}")
            
        # Create metadata message
        metadata = ShardChunk.Metadata(
            shard=shard.to_proto(),
            inference_engine_name=inference_engine_name,
            total_size=0,  # Will be set by sender
            file_name=""   # Will be set by sender
        )
        
        # Start transfer stream
        stream = peer.file_service.TransferShard()
        
        # Send initial metadata request
        if DEBUG >= 2:
            print(f"Sending metadata request for shard {shard}")
        await stream.send(ShardChunk(metadata=metadata))
        
        # Get response with actual metadata
        response = await stream.recv()
        if not response or response.status == TransferStatus.ERROR:
            error_msg = f"Failed to start transfer: {response.error_message if response else 'No response'}"
            if DEBUG >= 2:
                print(error_msg)
            raise Exception(error_msg)
            
        # Create temporary file to write chunks
        temp_path = Path(f"/tmp/shard_download_{shard.model_id}_{shard.start_layer}_{shard.end_layer}")
        if DEBUG >= 2:
            print(f"Writing shard {shard} to temporary file {temp_path}")
        
        try:
            with open(temp_path, "wb") as f:
                async for chunk in stream:
                    if chunk.HasField("chunk_data"):
                        f.write(chunk.chunk_data)
                        if DEBUG >= 3:  # More verbose logging for actual chunks
                            print(f"Received chunk of size {len(chunk.chunk_data)} at offset {chunk.offset}")
                        # Trigger progress callback
                        self._on_progress.trigger_all(
                            shard,
                            RepoProgressEvent(
                                bytes_processed=chunk.offset + len(chunk.chunk_data),
                                total_bytes=metadata.total_size
                            )
                        )
                        
                    if chunk.is_last:
                        if DEBUG >= 2:
                            print(f"Received last chunk for shard {shard}")
                        break
                        
            if DEBUG >= 2:
                print(f"Completed download of shard {shard} to {temp_path}")
            return temp_path
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"Error downloading shard {shard}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise e

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self._on_progress

    async def get_shard_download_status(self) -> Optional[Dict[str, float]]:
        if not self.current_shard:
            return None
            
        # Check if shard is already downloaded
        if self.current_shard in self.completed_downloads:
            return {"overall": 100.0}
            
        # Check active downloads
        if self.current_shard in self.active_downloads:
            # Status is tracked via progress callbacks
            # Return None to indicate in-progress but unknown percentage
            return None
            
        return None 