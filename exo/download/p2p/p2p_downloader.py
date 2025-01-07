import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import grpc
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
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

class P2PShardDownloader(ShardDownloader):
    def __init__(self, peers: List[PeerHandle], quick_check: bool = False):
        self.peers = peers
        self.quick_check = quick_check
        self.active_downloads: Dict[Shard, asyncio.Task] = {}
        self.completed_downloads: Dict[Shard, Path] = {}
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()
        self.current_shard: Optional[Shard] = None

    async def _check_peer_status(self, peer: PeerHandle, shard: Shard, inference_engine_name: str) -> Optional[GetShardStatusResponse]:
        """Check peer status with retries"""
        for attempt in range(MAX_RETRIES):
            try:
                status = await peer.file_service.GetShardStatus(
                    GetShardStatusRequest(
                        shard=shard.to_proto(),
                        inference_engine_name=inference_engine_name
                    )
                )
                return status
            except grpc.aio.AioRpcError as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Attempt {attempt + 1}/{MAX_RETRIES} failed for peer {peer}: {e.details()}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Unexpected error checking peer {peer}: {e}")
                break
        return None

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        self.current_shard = shard
        
        if shard in self.completed_downloads:
            if DEBUG >= 2:
                print(f"[P2P Download] Shard {shard} already downloaded at {self.completed_downloads[shard]}")
            return self.completed_downloads[shard]

        if shard in self.active_downloads:
            if DEBUG >= 2:
                print(f"[P2P Download] Download already in progress for shard {shard}")
            return await self.active_downloads[shard]

        # Find peers that have this shard
        available_peers = []
        if DEBUG >= 2:
            print(f"[P2P Download] Searching for peers with shard {shard}")
            
        for peer in self.peers:
            status = await self._check_peer_status(peer, shard, inference_engine_name)
            if status and status.has_shard:
                if DEBUG >= 2:
                    print(f"[P2P Download] Peer {peer} has shard {shard} at {status.local_path} (size: {status.file_size})")
                available_peers.append((peer, status))

        if not available_peers:
            if DEBUG >= 2:
                print(f"[P2P Download] No peers found with shard {shard}")
            raise FileNotFoundError(f"No peers have shard {shard}")

        # Sort peers by response time/reliability (future improvement)
        chosen_peer, status = available_peers[0]
        if DEBUG >= 2:
            print(f"[P2P Download] Selected peer {chosen_peer} to download shard {shard}")
        
        # Start download with retries
        for attempt in range(MAX_RETRIES):
            try:
                download_task = asyncio.create_task(
                    self._download_shard(shard, inference_engine_name, chosen_peer)
                )
                self.active_downloads[shard] = download_task
                
                path = await download_task
                self.completed_downloads[shard] = path
                if DEBUG >= 2:
                    print(f"[P2P Download] Successfully downloaded shard {shard} to {path}")
                return path
                
            except grpc.aio.AioRpcError as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e.details()}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise  # Re-raise on last attempt
                
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Unexpected error downloading shard: {e}")
                raise
                
            finally:
                if shard in self.active_downloads:
                    self.active_downloads.pop(shard)

    async def _download_shard(
        self, shard: Shard, inference_engine_name: str, peer: PeerHandle
    ) -> Path:
        if DEBUG >= 2:
            print(f"[P2P Download] Starting download of shard {shard} from peer {peer}")
            
        metadata = ShardChunk.Metadata(
            shard=shard.to_proto(),
            inference_engine_name=inference_engine_name,
            total_size=0,
            file_name=""
        )
        
        temp_path = Path(f"/tmp/shard_download_{shard.model_id}_{shard.start_layer}_{shard.end_layer}")
        
        try:
            # Start transfer stream with timeout
            async with asyncio.timeout(30):  # 30 second timeout for initial connection
                stream = peer.file_service.TransferShard()
                
                if DEBUG >= 2:
                    print(f"[P2P Download] Sending metadata request for shard {shard}")
                await stream.send(ShardChunk(metadata=metadata))
                
                response = await stream.recv()
                if not response or response.status == TransferStatus.ERROR:
                    error_msg = f"Failed to start transfer: {response.error_message if response else 'No response'}"
                    if DEBUG >= 2:
                        print(f"[P2P Download] {error_msg}")
                    raise Exception(error_msg)
            
            if DEBUG >= 2:
                print(f"[P2P Download] Writing shard {shard} to temporary file {temp_path}")
            
            with open(temp_path, "wb") as f:
                async with asyncio.timeout(300):  # 5 minute timeout for full transfer
                    async for chunk in stream:
                        if chunk.HasField("chunk_data"):
                            f.write(chunk.chunk_data)
                            if DEBUG >= 3:
                                print(f"[P2P Download] Received chunk of size {len(chunk.chunk_data)} at offset {chunk.offset}")
                            self._on_progress.trigger_all(
                                shard,
                                RepoProgressEvent(
                                    bytes_processed=chunk.offset + len(chunk.chunk_data),
                                    total_bytes=metadata.total_size
                                )
                            )
                            
                        if chunk.is_last:
                            if DEBUG >= 2:
                                print(f"[P2P Download] Received last chunk for shard {shard}")
                            break
                    
            if DEBUG >= 2:
                print(f"[P2P Download] Completed download of shard {shard} to {temp_path}")
            return temp_path
            
        except asyncio.TimeoutError as e:
            if DEBUG >= 2:
                print(f"[P2P Download] Timeout downloading shard {shard}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"[P2P Download] Error downloading shard {shard}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self._on_progress

    async def get_shard_download_status(self) -> Optional[Dict[str, float]]:
        if not self.current_shard:
            return None
            
        if self.current_shard in self.completed_downloads:
            return {"overall": 100.0}
            
        if self.current_shard in self.active_downloads:
            return None
            
        return None 