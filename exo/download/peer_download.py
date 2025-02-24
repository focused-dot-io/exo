from exo.inference.shard import Shard
from exo.models import get_repo
from pathlib import Path
from exo.download.shard_download import ShardDownloader
from exo.download.new_shard_download import (
    NewShardDownloader, SingletonShardDownloader, CachedShardDownloader,
    exo_home, ensure_downloads_dir, resolve_allow_patterns, get_downloaded_size,
    download_shard, fetch_file_list_with_cache
)
from exo.download.download_progress import RepoProgressEvent, RepoFileProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.networking.peer_handle import PeerHandle
from typing import List, Dict, Tuple, Optional, AsyncIterator, Set
import asyncio
import aiofiles
import aiofiles.os as aios
import time
from datetime import timedelta
import os
import json
import traceback


class PeerShardDownloader(ShardDownloader):
    """A ShardDownloader that first tries to download from peers, and if that fails, 
    downloads from HuggingFace directly. It also seeds downloaded models to other peers."""
    
    def __init__(self, fallback_downloader: ShardDownloader, max_parallel_downloads: int = 8):
        self.fallback_downloader = fallback_downloader
        self.max_parallel_downloads = max_parallel_downloads
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()
        self.peers: List[PeerHandle] = []
        self._coordinator_id = None
        self.downloading_models: Set[str] = set()
        
    async def _wait_for_model_on_coordinator(self, coordinator_peer, repo_id, revision="main", 
                                         max_wait_seconds=60, poll_interval_seconds=1.0):
        """Wait for the coordinator node to download a model
        
        Args:
            coordinator_peer: The peer handle for the coordinator node
            repo_id: The repo ID to wait for
            revision: The model revision to wait for (default: main)
            max_wait_seconds: Maximum time to wait in seconds (default: 60)
            poll_interval_seconds: How often to check if model is available (default: 1.0)
            
        Returns:
            bool: True if model was found on coordinator, False if timed out
        """
        print(f"[PEER DOWNLOAD] Waiting for coordinator {coordinator_peer.id()} to download model {repo_id}")
        
        start_time = time.time()
        attempts = 0
        
        while time.time() - start_time < max_wait_seconds:
            attempts += 1
            try:
                if not await coordinator_peer.is_connected():
                    print(f"[PEER DOWNLOAD] Warning: Coordinator peer {coordinator_peer.id()} is not connected")
                    await asyncio.sleep(poll_interval_seconds)
                    continue
                    
                # Poll to see if coordinator has the model now
                has_model_response = await coordinator_peer.has_model(repo_id, revision)
                
                if has_model_response.has_model:
                    complete_status = "complete" if has_model_response.is_complete else "incomplete"
                    print(f"[PEER DOWNLOAD] Found model {repo_id} on coordinator (status: {complete_status})")
                    return True
                    
                # If we're debugging, log poll attempts
                if DEBUG >= 2 or attempts % 5 == 0:  # Log every 5 attempts
                    elapsed = time.time() - start_time
                    print(f"[PEER DOWNLOAD] Waiting for model {repo_id} on coordinator (elapsed: {elapsed:.1f}s, attempt: {attempts})")
                    
            except Exception as e:
                print(f"[PEER DOWNLOAD] Error checking for model on coordinator: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
            
            # Wait before polling again
            await asyncio.sleep(poll_interval_seconds)
        
        print(f"[PEER DOWNLOAD] Timed out waiting for coordinator to download model {repo_id} after {max_wait_seconds} seconds")
        return False
        
    @staticmethod
    def filter_repo_objects(objects, allow_patterns=None, key=None):
        """Filter repository objects based on allow patterns"""
        if not allow_patterns:
            return objects
            
        if key is None:
            key = lambda x: x
            
        for obj in objects:
            path = key(obj)
            if any(path.startswith(pattern.strip("*")) for pattern in allow_patterns) or "*" in allow_patterns:
                yield obj
        
    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self._on_progress
    
    def set_peers(self, peers: List[PeerHandle]):
        """Set the list of available peers"""
        self.peers = peers
        
    def set_coordinator_id(self, coordinator_id: str):
        """Set the ID of the coordinator node"""
        self._coordinator_id = coordinator_id
    
    async def find_peer_with_model(self, repo_id: str, revision: str = "main") -> Optional[PeerHandle]:
        """Find a peer that has the model we're looking for"""
        print(f"[PEER DOWNLOAD] Searching for peer with model: {repo_id}")
            
        best_peer = None
        most_complete = False
        
        if not self.peers:
            print(f"[PEER DOWNLOAD] No peers available to search for model {repo_id}")
            return None
            
        print(f"[PEER DOWNLOAD] Checking {len(self.peers)} peers for model {repo_id}")
        
        for i, peer in enumerate(self.peers):
            try:
                if await peer.is_connected():
                    print(f"[PEER DOWNLOAD] Checking peer {i+1}/{len(self.peers)}: {peer.id()}")
                    # This would call the new HasModel RPC
                    has_model_response = await peer.has_model(repo_id, revision)
                    if has_model_response.has_model:
                        if has_model_response.is_complete:
                            print(f"[PEER DOWNLOAD] Found peer {peer.id()} with complete model {repo_id}")
                            return peer
                        elif not most_complete:
                            print(f"[PEER DOWNLOAD] Found peer {peer.id()} with partial model {repo_id}")
                            best_peer = peer
                    else:
                        print(f"[PEER DOWNLOAD] Peer {peer.id()} does not have model {repo_id}")
                else:
                    print(f"[PEER DOWNLOAD] Peer {peer.id()} is not connected")
            except Exception as e:
                print(f"[PEER DOWNLOAD] Error checking if peer {peer.id()} has model {repo_id}: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
                    
        if best_peer:
            print(f"[PEER DOWNLOAD] Best peer found is {best_peer.id()} (incomplete model)")
        else:
            print(f"[PEER DOWNLOAD] No peers found with model {repo_id}")
            
        return best_peer
    
    async def download_file_from_peer(
        self, 
        peer: PeerHandle, 
        repo_id: str, 
        revision: str, 
        file_path: str, 
        target_dir: Path,
        on_progress: callable
    ) -> Path:
        """Download a single file from a peer"""
        print(f"[PEER DOWNLOAD] Downloading file {file_path} from peer {peer.id()}")
            
        try:
            # Create directory structure
            await aios.makedirs((target_dir/file_path).parent, exist_ok=True)
            
            # Check if file already exists
            if await aios.path.exists(target_dir/file_path):
                print(f"[PEER DOWNLOAD] File {file_path} already exists, skipping")
                return target_dir/file_path
                
            partial_path = target_dir/f"{file_path}.partial"
            resume_byte_pos = (await aios.stat(partial_path)).st_size if (await aios.path.exists(partial_path)) else 0
            
            if resume_byte_pos > 0:
                print(f"[PEER DOWNLOAD] Resuming download of {file_path} from byte position {resume_byte_pos}")
            
            # This would call the new GetModelFile RPC to stream file chunks
            file_start_time = time.time()
            total_size = 0
            downloaded = resume_byte_pos
            
            async with aiofiles.open(partial_path, 'ab' if resume_byte_pos else 'wb') as f:
                async for chunk in peer.get_model_file(repo_id, revision, file_path, resume_byte_pos):
                    chunk_size = len(chunk.data)
                    downloaded += await f.write(chunk.data)
                    total_size = chunk.total_size
                    on_progress(downloaded, total_size)
                    
                    # Print progress occasionally but not for every chunk
                    if DEBUG >= 3 or (downloaded % (10 * 1024 * 1024) < chunk_size): # Print every ~10MB
                        progress_pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                        print(f"[PEER DOWNLOAD] Progress: {downloaded/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB ({progress_pct:.1f}%)")
            
            file_download_time = time.time() - file_start_time
            download_speed_mbps = (downloaded - resume_byte_pos) / 1024 / 1024 / file_download_time if file_download_time > 0 else 0
            
            # Rename from partial to final
            await aios.rename(partial_path, target_dir/file_path)
            
            print(f"[PEER DOWNLOAD] Successfully downloaded {file_path} ({downloaded/1024/1024:.2f} MB) in {file_download_time:.2f}s ({download_speed_mbps:.2f} MB/s)")
            return target_dir/file_path
            
        except Exception as e:
            print(f"[PEER DOWNLOAD] Error downloading file {file_path} from peer {peer.id()}: {e}")
            if DEBUG >= 2:
                traceback.print_exc()
            raise e
    
    async def download_model_from_peer(
        self, 
        peer: PeerHandle, 
        shard: Shard, 
        inference_engine_name: str, 
        on_progress: AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]
    ) -> Path:
        """Download a full model from a peer"""
        repo_id = get_repo(shard.model_id, inference_engine_name)
        revision = "main"
        target_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
        await aios.makedirs(target_dir, exist_ok=True)
        
        # Get list of files from peer
        allow_patterns = await resolve_allow_patterns(shard, inference_engine_name)
        file_list_response = await peer.get_model_file_list(repo_id, revision, allow_patterns)
        
        if not file_list_response.files:
            raise Exception(f"Peer {peer.id()} returned empty file list for {repo_id}")
            
        # Convert the file list to the format our progress tracking expects
        file_list = [{"path": f.path, "size": f.size} for f in file_list_response.files]
        
        # Setup progress tracking (similar to download_shard in new_shard_download.py)
        all_start_time = time.time()
        file_progress: Dict[str, RepoFileProgressEvent] = {}
        
        # Setup progress callback
        def on_progress_wrapper(file: dict, curr_bytes: int, total_bytes: int):
            start_time = file_progress[file["path"]].start_time if file["path"] in file_progress else time.time()
            downloaded_this_session = file_progress[file["path"]].downloaded_this_session + (curr_bytes - file_progress[file["path"]].downloaded) if file["path"] in file_progress else curr_bytes
            speed = downloaded_this_session / (time.time() - start_time) if time.time() - start_time > 0 else 0
            eta = timedelta(seconds=(total_bytes - curr_bytes) / speed) if speed > 0 else timedelta(seconds=0)
            file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], curr_bytes, downloaded_this_session, total_bytes, speed, eta, "complete" if curr_bytes == total_bytes else "in_progress", start_time)
            
            from exo.download.new_shard_download import calculate_repo_progress
            on_progress.trigger_all(shard, calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time))
            
        # Initialize progress tracking
        for file in file_list:
            downloaded_bytes = await get_downloaded_size(target_dir/file["path"])
            file_progress[file["path"]] = RepoFileProgressEvent(repo_id, revision, file["path"], downloaded_bytes, 0, file["size"], 0, timedelta(0), "complete" if downloaded_bytes == file["size"] else "not_started", time.time())
        
        # Download files in parallel
        semaphore = asyncio.Semaphore(self.max_parallel_downloads)
        
        async def download_with_semaphore(file):
            async with semaphore:
                await self.download_file_from_peer(
                    peer, 
                    repo_id, 
                    revision, 
                    file["path"], 
                    target_dir,
                    lambda curr_bytes, total_bytes: on_progress_wrapper(file, curr_bytes, total_bytes)
                )
                
        await asyncio.gather(*[download_with_semaphore(file) for file in file_list])
        
        # Calculate final progress
        from exo.download.new_shard_download import calculate_repo_progress
        final_repo_progress = calculate_repo_progress(shard, repo_id, revision, file_progress, all_start_time)
        on_progress.trigger_all(shard, final_repo_progress)
        
        return target_dir
    
    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        """Ensure a shard is downloaded, preferably from a peer"""
        repo_id = get_repo(shard.model_id, inference_engine_name)
        
        if not repo_id:
            raise ValueError(f"No repo found for {shard.model_id} and inference engine {inference_engine_name}")
        
        # Check if we already have this model downloaded
        target_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
        if await aios.path.exists(target_dir):
            # Check if the model is complete
            allow_patterns = await resolve_allow_patterns(shard, inference_engine_name)
            file_list = await fetch_file_list_with_cache(repo_id)
            
            # Use the class method for filtering
            filtered_file_list = list(self.filter_repo_objects(file_list, allow_patterns=allow_patterns, key=lambda x: x["path"]))
            
            all_files_exist = True
            for file in filtered_file_list:
                if not await aios.path.exists(target_dir/file["path"]):
                    all_files_exist = False
                    break
                    
            if all_files_exist:
                return target_dir
        
        # If I have no peers, download directly
        if not self.peers:
            print(f"[PEER DOWNLOAD] No peers available, downloading {repo_id} directly")
            return await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
            
        # If I'm the coordinator, download directly
        if self._coordinator_id and self.peers and self._coordinator_id == self.peers[0].id():
            print(f"[PEER DOWNLOAD] I am the coordinator node ({self._coordinator_id}), downloading {repo_id} directly")
            return await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
            
        # Otherwise I should try to download from peers
        print(f"[PEER DOWNLOAD] I am NOT the coordinator. My ID: {self.peers[0].id()}, Coordinator: {self._coordinator_id}")
        print(f"[PEER DOWNLOAD] Will try to download {repo_id} from peers")
        
        # Find the coordinator peer
        coordinator_peer = None
        for peer in self.peers:
            if peer.id() == self._coordinator_id:
                coordinator_peer = peer
                break
                
        if coordinator_peer:
            # First check if coordinator already has the model
            has_model_response = await coordinator_peer.has_model(repo_id)
            coordinator_has_model = has_model_response.has_model
            
            if not coordinator_has_model:
                # Wait for coordinator to download the model
                print(f"[PEER DOWNLOAD] Coordinator does not have model {repo_id} yet, waiting for it to download...")
                
                # Use our wait method to poll for coordinator to complete download
                wait_success = await self._wait_for_model_on_coordinator(coordinator_peer, repo_id)
                
                if wait_success:
                    print(f"[PEER DOWNLOAD] Coordinator now has model {repo_id}, will download from coordinator")
        
        # Try to find a peer that has this model (preferably the coordinator)
        peer = await self.find_peer_with_model(repo_id)
        
        if peer:
            try:
                print(f"[PEER DOWNLOAD] Found peer {peer.id()} that has model {repo_id}")
                print(f"[PEER DOWNLOAD] Starting download of {repo_id} from peer {peer.id()}")
                    
                # Track that we're downloading this model
                self.downloading_models.add(repo_id)
                
                # Download from peer
                download_start = time.time()
                target_dir = await self.download_model_from_peer(
                    peer, 
                    shard, 
                    inference_engine_name, 
                    self.on_progress
                )
                download_time = time.time() - download_start
                
                # No longer downloading this model
                self.downloading_models.remove(repo_id)
                
                print(f"[PEER DOWNLOAD] Successfully downloaded {repo_id} from peer {peer.id()} in {download_time:.2f} seconds")
                return target_dir
                
            except Exception as e:
                print(f"[PEER DOWNLOAD] Failed to download {repo_id} from peer {peer.id()}, falling back to direct download")
                print(f"[PEER DOWNLOAD] Error: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
                    
                # No longer downloading this model
                if repo_id in self.downloading_models:
                    self.downloading_models.remove(repo_id)
        
        # If we got here, either no peer had the model or download from peer failed
        # Fall back to direct download
        print(f"[PEER DOWNLOAD] No peers have {repo_id} or peer download failed")
        print(f"[PEER DOWNLOAD] Falling back to direct download from HuggingFace")
            
        direct_download_start = time.time()
        result = await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
        direct_download_time = time.time() - direct_download_start
        
        print(f"[PEER DOWNLOAD] Direct download of {repo_id} completed in {direct_download_time:.2f} seconds")
        return result
    
    async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
        """Get the download status of all shards"""
        async for path, status in self.fallback_downloader.get_shard_download_status(inference_engine_name):
            yield path, status


def peer_shard_downloader(max_parallel_downloads: int = 8) -> ShardDownloader:
    """Create a new PeerShardDownloader with fallback to the standard downloader"""
    fallback = SingletonShardDownloader(CachedShardDownloader(NewShardDownloader(max_parallel_downloads)))
    return PeerShardDownloader(fallback, max_parallel_downloads)