import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import timedelta
import tempfile
import shutil
import grpc
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.networking.peer_handle import PeerHandle
from exo.networking.grpc.node_service_pb2 import (
    GetShardStatusRequest, GetShardStatusResponse,
    ShardChunk, TransferStatus
)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
CONNECT_TIMEOUT = 10.0  # seconds
TRANSFER_TIMEOUT = 300.0  # seconds

class PeerConnectionError(Exception):
    """Raised when a peer connection fails"""
    pass

class P2PShardDownloader(ShardDownloader):
    def __init__(self, peers: List[PeerHandle], quick_check: bool = False):
        self.peers = peers
        self.quick_check = quick_check
        self.active_downloads: Dict[Shard, asyncio.Task] = {}
        self.completed_downloads: Dict[Shard, Path] = {}
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()
        self.current_shard: Optional[Shard] = None
        self.failed_peers: Set[PeerHandle] = set()

    async def _check_peer_status(self, peer: PeerHandle, shard: Shard, inference_engine_name: str) -> Optional[GetShardStatusResponse]:
        """Check peer status with retries"""
        if peer in self.failed_peers:
            if DEBUG >= 2:
                print(f"[P2P Download] Skipping previously failed peer {peer}")
            return None

        if DEBUG >= 2:
            print(f"[P2P Download] Checking if peer {peer} has shard {shard}")
            print(f"[P2P Download] Peer details: {peer}")
        
        # Map of GRPC channel states for debugging
        GRPC_STATES = {
            0: "IDLE",
            1: "CONNECTING",
            2: "READY",
            3: "TRANSIENT_FAILURE",
            4: "SHUTDOWN"
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                is_connected = await peer.is_connected()
                if DEBUG >= 2:
                    print(f"[P2P Download] Peer connection status: {is_connected}")
                    if hasattr(peer, 'channel') and peer.channel:
                        state = peer.channel._channel.check_connectivity_state(True)
                        print(f"[P2P Download] Current channel state: {GRPC_STATES.get(state, f'UNKNOWN({state})')}")
                
                if not is_connected:
                    if DEBUG >= 2:
                        print(f"[P2P Download] Attempting to connect to peer {peer} (attempt {attempt + 1})")
                    try:
                        async with asyncio.timeout(CONNECT_TIMEOUT):
                            await peer.connect()
                            
                            # Verify channel exists after connect
                            if not hasattr(peer, 'channel') or not peer.channel:
                                raise PeerConnectionError("Channel not created after connect")
                            
                            # Wait for channel to be ready
                            state = peer.channel._channel.check_connectivity_state(True)
                            if DEBUG >= 2:
                                print(f"[P2P Download] Initial connection state: {GRPC_STATES.get(state, f'UNKNOWN({state})')}")
                            
                            # Wait for READY state with timeout
                            ready_timeout = CONNECT_TIMEOUT / 2  # Use half the connect timeout for ready wait
                            async with asyncio.timeout(ready_timeout):
                                while state != 2:  # READY
                                    if state == 4:  # SHUTDOWN
                                        raise PeerConnectionError("Channel shutdown while waiting for ready state")
                                    if state == 3:  # TRANSIENT_FAILURE
                                        raise PeerConnectionError("Channel in transient failure while waiting for ready state")
                                        
                                    await peer.channel.channel_ready()
                                    state = peer.channel._channel.check_connectivity_state(True)
                                    if DEBUG >= 2:
                                        print(f"[P2P Download] Channel state changed to: {GRPC_STATES.get(state, f'UNKNOWN({state})')}")
                                
                    except (asyncio.TimeoutError, asyncio.CancelledError, grpc.aio.AioRpcError, PeerConnectionError) as e:
                        if DEBUG >= 2:
                            print(f"[P2P Download] Connection error for peer {peer}: {str(e)}")
                            print(f"[P2P Download] Error type: {type(e)}")
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                            continue
                        self.failed_peers.add(peer)
                        return None

                # Try to get shard status
                async with asyncio.timeout(CONNECT_TIMEOUT):
                    if DEBUG >= 2:
                        print(f"[P2P Download] Requesting shard status from peer {peer}")
                    status = await peer.stub.GetShardStatus(
                        GetShardStatusRequest(
                            shard=shard.to_proto(),
                            inference_engine_name=inference_engine_name
                        )
                    )
                    if DEBUG >= 2:
                        print(f"[P2P Download] Got status from peer {peer}: has_shard={getattr(status, 'has_shard', False)}")
                        if getattr(status, 'has_shard', False):
                            print(f"[P2P Download] Shard location: {getattr(status, 'local_path', 'unknown')}, size: {getattr(status, 'file_size', 0)}")
                    return status
                    
            except (asyncio.TimeoutError, asyncio.CancelledError, grpc.aio.AioRpcError) as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Error checking peer {peer} (attempt {attempt + 1}): {str(e)}")
                    print(f"[P2P Download] Error type: {type(e)}")
                if attempt == MAX_RETRIES - 1:
                    self.failed_peers.add(peer)
            
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

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
            if peer in self.failed_peers:
                continue
                
            status = await self._check_peer_status(peer, shard, inference_engine_name)
            if status:
                if DEBUG >= 2:
                    print(f"[P2P Download] Peer {peer} has shard {shard} at {status.local_path} (size: {status.file_size})")
                available_peers.append((peer, status))

        if not available_peers:
            if DEBUG >= 2:
                print(f"[P2P Download] No peers found with shard {shard}")
            raise FileNotFoundError(f"No peers have shard {shard}")

        # Try each available peer until one succeeds
        last_error = None
        for peer, status in available_peers:
            if peer in self.failed_peers:
                continue
                
            try:
                if DEBUG >= 2:
                    print(f"[P2P Download] Attempting download from peer {peer}")
                    
                download_task = asyncio.create_task(
                    self._download_shard(shard, inference_engine_name, peer)
                )
                self.active_downloads[shard] = download_task
                
                path = await download_task
                self.completed_downloads[shard] = path
                if DEBUG >= 2:
                    print(f"[P2P Download] Successfully downloaded shard {shard} to {path}")
                return path
                
            except (asyncio.TimeoutError, grpc.aio.AioRpcError) as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Failed to download from peer {peer}: {str(e)}")
                self.failed_peers.add(peer)
                last_error = e
                continue
                
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Unexpected error downloading from peer {peer}: {e}")
                self.failed_peers.add(peer)
                last_error = e
                continue
                
            finally:
                if shard in self.active_downloads:
                    self.active_downloads.pop(shard)

        # If we get here, all peers failed
        raise PeerConnectionError(f"All peers failed to download shard {shard}. Last error: {last_error}")

    async def _download_shard(
        self, shard: Shard, inference_engine_name: str, peer: PeerHandle
    ) -> Path:
        if DEBUG >= 2:
            print(f"[P2P Download] Starting download of shard {shard} from peer {peer}")
            
        try:
            metadata = ShardChunk()
            metadata.metadata.shard.CopyFrom(shard.to_proto())
            metadata.metadata.inference_engine_name = inference_engine_name
        except Exception as e:
            if DEBUG >= 2:
                print(f"[P2P Download] Error creating metadata: {e}")
            raise
        
        # Create a fresh temporary directory
        temp_dir = None
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="shard_download_"))
            model_dir = temp_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            temp_path = model_dir / "model.safetensors"
            
            if DEBUG >= 2:
                print(f"[P2P Download] Created temporary directory at {temp_dir}")
            
            # Start transfer stream with timeout
            async with asyncio.timeout(CONNECT_TIMEOUT):
                if DEBUG >= 2:
                    print(f"[P2P Download] Starting transfer stream for shard {shard}")
                
                # Initialize stream and send metadata request
                stream = peer.stub.TransferShard()
                await stream.write(metadata)
                
                if DEBUG >= 2:
                    print(f"[P2P Download] Writing shard {shard} to temporary file {temp_path}")
                
                # Report initial progress
                self._on_progress.trigger_all(shard, RepoProgressEvent(
                    repo_id=shard.model_id,
                    repo_revision="main",
                    completed_files=0,
                    total_files=1,
                    downloaded_bytes=0,
                    downloaded_bytes_this_session=0,
                    total_bytes=0,  # Will be updated with first chunk
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    file_progress={},
                    status="downloading"
                ))
                
                with open(temp_path, "wb") as f:
                    start_time = asyncio.get_event_loop().time()
                    async with asyncio.timeout(TRANSFER_TIMEOUT):
                        # Read the initial metadata response
                        initial_response = await stream.read()
                        total_size = 0
                        
                        if DEBUG >= 2:
                            print(f"[P2P Download] Got initial response: {initial_response}")
                        
                        if isinstance(initial_response, TransferStatus):
                            try:
                                total_size = initial_response.file_size  # Changed from total_size to file_size
                                if DEBUG >= 2:
                                    print(f"[P2P Download] Got initial status with file size: {total_size}")
                            except Exception as e:
                                if DEBUG >= 2:
                                    print(f"[P2P Download] Error getting file size from status: {e}")
                                total_size = 0
                        
                        while True:
                            chunk = await stream.read()
                            if not chunk:
                                if DEBUG >= 2:
                                    print("[P2P Download] Stream ended")
                                break
                                
                            if not isinstance(chunk, ShardChunk):
                                if DEBUG >= 2:
                                    print(f"[P2P Download] Unexpected message type: {type(chunk)}")
                                continue
                                
                            chunk_data = getattr(chunk, 'chunk_data', None)
                            if chunk_data:
                                f.write(chunk_data)
                                offset = getattr(chunk, 'offset', 0)
                                bytes_processed = offset + len(chunk_data)
                                total_bytes = total_size or bytes_processed
                                
                                if DEBUG >= 3:
                                    print(f"[P2P Download] Received chunk of size {len(chunk_data)} at offset {offset}")
                                
                                # Calculate speed and ETA
                                elapsed = asyncio.get_event_loop().time() - start_time
                                speed = bytes_processed / elapsed if elapsed > 0 else 0
                                remaining_bytes = total_bytes - bytes_processed
                                eta = remaining_bytes / speed if speed > 0 else 0
                                
                                self._on_progress.trigger_all(
                                    shard,
                                    RepoProgressEvent(
                                        repo_id=shard.model_id,
                                        repo_revision="main",
                                        completed_files=0,
                                        total_files=1,
                                        downloaded_bytes=bytes_processed,
                                        downloaded_bytes_this_session=bytes_processed,
                                        total_bytes=total_bytes,
                                        overall_speed=speed,
                                        overall_eta=timedelta(seconds=eta),
                                        file_progress={},
                                        status="downloading"
                                    )
                                )
                            
                            if getattr(chunk, 'is_last', False):
                                if DEBUG >= 2:
                                    print(f"[P2P Download] Received last chunk for shard {shard}")
                                break
                    
                    # Report completion
                    self._on_progress.trigger_all(shard, RepoProgressEvent(
                        repo_id=shard.model_id,
                        repo_revision="main",
                        completed_files=1,
                        total_files=1,
                        downloaded_bytes=total_size,
                        downloaded_bytes_this_session=total_size,
                        total_bytes=total_size,
                        overall_speed=0,
                        overall_eta=timedelta(seconds=0),
                        file_progress={},
                        status="complete"
                    ))
                    
            if DEBUG >= 2:
                print(f"[P2P Download] Completed download of shard {shard} to {temp_path}")
            return temp_path
            
        except asyncio.TimeoutError as e:
            if DEBUG >= 2:
                print(f"[P2P Download] Timeout downloading shard {shard}: {e}")
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            # Report failure
            self._on_progress.trigger_all(shard, RepoProgressEvent(
                repo_id=shard.model_id,
                repo_revision="main",
                completed_files=0,
                total_files=1,
                downloaded_bytes=0,
                downloaded_bytes_this_session=0,
                total_bytes=0,
                overall_speed=0,
                overall_eta=timedelta(seconds=0),
                file_progress={},
                status="failed"
            ))
            raise
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"[P2P Download] Error downloading shard {shard}: {e}")
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            # Report failure
            self._on_progress.trigger_all(shard, RepoProgressEvent(
                repo_id=shard.model_id,
                repo_revision="main",
                completed_files=0,
                total_files=1,
                downloaded_bytes=0,
                downloaded_bytes_this_session=0,
                total_bytes=0,
                overall_speed=0,
                overall_eta=timedelta(seconds=0),
                file_progress={},
                status="failed"
            ))
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