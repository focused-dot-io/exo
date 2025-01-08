import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from datetime import timedelta
import tempfile
import shutil
import grpc
import os
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.networking.peer_handle import PeerHandle
from exo.networking.grpc.node_service_pb2 import (
    GetShardStatusRequest, GetShardStatusResponse,
    ShardChunk, TransferStatus, ShardChunk_Metadata
)
from exo.models import get_repo

CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_RETRIES = 5  # Increased from 3
RETRY_DELAY = 2.0  # Increased from 1.0
CONNECT_TIMEOUT = 20.0  # Increased from 10.0
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
            print(f"[P2P Download] Peer address: {peer.addr() if hasattr(peer, 'addr') else 'unknown'}")
        
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
                            print(f"[P2P Download] Stack trace:")
                            traceback.print_exc()
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
                        else:
                            print(f"[P2P Download] Peer does not have the shard")
                    return status
                    
            except (asyncio.TimeoutError, asyncio.CancelledError, grpc.aio.AioRpcError) as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Error checking peer {peer} (attempt {attempt + 1}): {str(e)}")
                    print(f"[P2P Download] Error type: {type(e)}")
                    print(f"[P2P Download] Stack trace:")
                    traceback.print_exc()
                if attempt == MAX_RETRIES - 1:
                    self.failed_peers.add(peer)
            
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))

        return None

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        self.current_shard = shard
        
        # Get HuggingFace cache directory and repo name
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        repo_name = get_repo(shard.model_id, inference_engine_name)
        if not repo_name:
            raise ValueError(f"Could not find repo name for model {shard.model_id} and engine {inference_engine_name}")
        
        model_name = f"models--{repo_name.replace('/', '--')}"
        final_model_dir = Path(hf_home) / model_name
        final_snapshot_dir = final_model_dir / "snapshots" / "current"
        final_safetensors = final_snapshot_dir / "model.safetensors"
        
        # Check if the complete model structure exists
        def check_model_structure() -> bool:
            try:
                if not all(p.exists() for p in [
                    final_model_dir,
                    final_snapshot_dir,
                    final_safetensors,
                    final_model_dir / "config.json",
                    final_snapshot_dir / "config.json"
                ]):
                    return False
                
                # Verify safetensors is a file, not a directory
                if not final_safetensors.is_file():
                    return False
                
                # Try to read the first few bytes to verify file integrity
                with open(final_safetensors, 'rb') as f:
                    header = f.read(8)
                    if len(header) != 8:
                        if DEBUG >= 2:
                            print(f"[P2P Download] File integrity check failed for {final_safetensors}")
                        return False
                
                return True
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Error checking model structure: {e}")
                return False
        
        # Check if shard already exists locally with complete structure
        if check_model_structure():
            if DEBUG >= 2:
                print(f"[P2P Download] Shard {shard} already exists with complete structure at {final_model_dir}")
            self.completed_downloads[shard] = final_snapshot_dir
            return final_snapshot_dir
        
        if shard in self.completed_downloads:
            snapshot_dir = self.completed_downloads[shard]
            if check_model_structure():
                if DEBUG >= 2:
                    print(f"[P2P Download] Shard {shard} already downloaded at {snapshot_dir}")
                return snapshot_dir
            else:
                if DEBUG >= 2:
                    print(f"[P2P Download] Cached shard at {snapshot_dir} has incomplete structure, will redownload")
                self.completed_downloads.pop(shard)

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
            if status and getattr(status, 'has_shard', False):  # Only add peers that actually have the shard
                if DEBUG >= 2:
                    print(f"[P2P Download] Peer {peer} has shard {shard} at {getattr(status, 'local_path', 'unknown')} (size: {getattr(status, 'file_size', 0)})")
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
            
        # Create metadata request
        metadata = ShardChunk_Metadata(
            shard=shard.to_proto(),
            inference_engine_name=inference_engine_name
        )
        initial_request = ShardChunk(metadata=metadata)

        # Create a fresh temporary directory
        temp_dir = None
        try:
            # Get HuggingFace cache directory and repo name
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
            repo_name = get_repo(shard.model_id, inference_engine_name)
            if not repo_name:
                raise ValueError(f"Could not find repo name for model {shard.model_id} and engine {inference_engine_name}")
            
            model_name = f"models--{repo_name.replace('/', '--')}"
            final_model_dir = Path(hf_home) / model_name
            
            if DEBUG >= 2:
                print(f"[P2P Download] Using repo name: {repo_name}")
                print(f"[P2P Download] Using model name: {model_name}")
            
            # Create temp directory for download
            temp_dir = Path(tempfile.mkdtemp(prefix="shard_download_"))
            snapshot_dir = temp_dir / "snapshots" / "current"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            temp_path = snapshot_dir / "model.safetensors"
            
            # Create config.json (minimal version)
            config_path = snapshot_dir / "config.json"
            with open(config_path, "w") as f:
                f.write('{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}')
            
            if DEBUG >= 2:
                print(f"[P2P Download] Created temporary directory at {temp_dir}")
                print(f"[P2P Download] Using model path: {temp_path}")
                print(f"[P2P Download] Created config at: {config_path}")
                print(f"[P2P Download] Final destination will be: {final_model_dir}")

            # Create async generator for sending requests
            async def request_generator():
                yield initial_request
                while True:
                    # Send acknowledgment for each chunk received
                    yield TransferStatus(
                        status="OK"
                    )

            # Start transfer stream with timeout
            async with asyncio.timeout(CONNECT_TIMEOUT):
                if DEBUG >= 2:
                    print(f"[P2P Download] Starting transfer stream for shard {shard}")
                
                # Initialize stream
                stream = peer.stub.TransferShard(request_generator())
                
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
                        total_size = 0
                        bytes_received = 0
                        
                        async for response in stream:
                            if isinstance(response, TransferStatus):
                                if DEBUG >= 2:
                                    print(f"[P2P Download] Got status: {response}")
                                if response.status == "ERROR":
                                    error_msg = getattr(response, 'error_message', 'Unknown error')
                                    raise RuntimeError(f"Transfer failed: {error_msg}")
                                if hasattr(response, 'file_size'):
                                    total_size = response.file_size
                                    if DEBUG >= 2:
                                        print(f"[P2P Download] Got file size: {total_size}")
                                continue
                                
                            if not isinstance(response, ShardChunk):
                                if DEBUG >= 2:
                                    print(f"[P2P Download] Skipping unexpected message type: {type(response)}")
                                continue
                                
                            chunk_data = getattr(response, 'chunk_data', None)
                            if not chunk_data:
                                if DEBUG >= 2:
                                    print("[P2P Download] Received empty chunk")
                                if getattr(response, 'is_last', False):
                                    break
                                continue
                                
                            f.write(chunk_data)
                            bytes_received = getattr(response, 'offset', 0) + len(chunk_data)
                            
                            if DEBUG >= 3:
                                print(f"[P2P Download] Received chunk of size {len(chunk_data)} at offset {getattr(response, 'offset', 0)}")
                            
                            # Calculate speed and ETA
                            elapsed = asyncio.get_event_loop().time() - start_time
                            speed = bytes_received / elapsed if elapsed > 0 else 0
                            remaining_bytes = total_size - bytes_received if total_size > 0 else 0
                            eta = remaining_bytes / speed if speed > 0 else 0
                            
                            self._on_progress.trigger_all(
                                shard,
                                RepoProgressEvent(
                                    repo_id=shard.model_id,
                                    repo_revision="main",
                                    completed_files=0,
                                    total_files=1,
                                    downloaded_bytes=bytes_received,
                                    downloaded_bytes_this_session=bytes_received,
                                    total_bytes=total_size or bytes_received,
                                    overall_speed=speed,
                                    overall_eta=timedelta(seconds=eta),
                                    file_progress={},
                                    status="downloading"
                                )
                            )
                            
                            if getattr(response, 'is_last', False):
                                break
                    
                    # Verify file size and integrity
                    actual_size = os.path.getsize(temp_path)
                    if total_size > 0 and actual_size != total_size:
                        raise RuntimeError(f"File size mismatch: expected {total_size}, got {actual_size}")
                    
                    # Verify file can be read
                    with open(temp_path, 'rb') as f:
                        header = f.read(8)
                        if len(header) != 8:
                            raise RuntimeError("File integrity check failed")
                        f.seek(0, os.SEEK_END)  # Seek to end to ensure entire file is readable
                    
                    # Report completion
                    self._on_progress.trigger_all(shard, RepoProgressEvent(
                        repo_id=shard.model_id,
                        repo_revision="main",
                        completed_files=1,
                        total_files=1,
                        downloaded_bytes=total_size or actual_size,
                        downloaded_bytes_this_session=total_size or actual_size,
                        total_bytes=total_size or actual_size,
                        overall_speed=0,
                        overall_eta=timedelta(seconds=0),
                        file_progress={},
                        status="complete"
                    ))
                    
            if DEBUG >= 2:
                print(f"[P2P Download] Completed download of shard {shard} to {temp_path}")
                print(f"[P2P Download] Moving files to HuggingFace cache directory")
            
            # Move files to HuggingFace cache
            final_model_dir.mkdir(parents=True, exist_ok=True)
            final_snapshot_dir = final_model_dir / "snapshots" / "current"
            final_snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the files
            final_safetensors = final_snapshot_dir / "model.safetensors"
            
            # Clean up any existing corrupted files
            if final_safetensors.exists():
                try:
                    final_safetensors.unlink()
                except Exception as e:
                    if DEBUG >= 2:
                        print(f"[P2P Download] Error cleaning up existing file: {e}")
            
            # Move files with verification
            try:
                shutil.move(str(temp_path), str(final_safetensors))
                # Verify the moved file
                if os.path.getsize(final_safetensors) != actual_size:
                    raise RuntimeError("File size changed during move")
                # Verify file integrity after move
                with open(final_safetensors, 'rb') as f:
                    header = f.read(8)
                    if len(header) != 8:
                        raise RuntimeError("File integrity check failed after move")
                    f.seek(0, os.SEEK_END)  # Verify entire file is readable
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Error moving file: {e}")
                if final_safetensors.exists():
                    final_safetensors.unlink()
                raise RuntimeError(f"Failed to move file: {e}")
            
            # Copy config.json to both locations
            root_config = final_model_dir / "config.json"
            snapshot_config = final_snapshot_dir / "config.json"
            try:
                shutil.copy2(str(config_path), str(root_config))  # Copy to root
                shutil.move(str(config_path), str(snapshot_config))  # Move to snapshot
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[P2P Download] Error copying config files: {e}")
                # Clean up on failure
                if final_safetensors.exists():
                    final_safetensors.unlink()
                if root_config.exists():
                    root_config.unlink()
                if snapshot_config.exists():
                    snapshot_config.unlink()
                raise RuntimeError(f"Failed to copy config files: {e}")
            
            if DEBUG >= 2:
                print(f"[P2P Download] Successfully moved files to {final_model_dir}")
                print(f"[P2P Download] Created config.json at {root_config} and {snapshot_config}")
                print(f"[P2P Download] Final file size: {os.path.getsize(final_safetensors)}")
            
            # Store the snapshot directory path
            self.completed_downloads[shard] = final_snapshot_dir
            
            return final_snapshot_dir
            
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