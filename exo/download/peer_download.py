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
        if DEBUG >= 2:
            print(f"Searching for peer with model: {repo_id}")
            
        best_peer = None
        most_complete = False
        
        for peer in self.peers:
            try:
                if await peer.is_connected():
                    # This would call the new HasModel RPC
                    has_model_response = await peer.has_model(repo_id, revision)
                    if has_model_response.has_model:
                        if has_model_response.is_complete:
                            return peer
                        elif not most_complete:
                            best_peer = peer
            except Exception as e:
                if DEBUG >= 1:
                    print(f"Error checking if peer {peer.id()} has model {repo_id}: {e}")
                    
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
        if DEBUG >= 2:
            print(f"Downloading {file_path} from peer {peer.id()}")
            
        try:
            # Create directory structure
            await aios.makedirs((target_dir/file_path).parent, exist_ok=True)
            
            # Check if file already exists
            if await aios.path.exists(target_dir/file_path):
                if DEBUG >= 3:
                    print(f"File {file_path} already exists, skipping")
                return target_dir/file_path
                
            partial_path = target_dir/f"{file_path}.partial"
            resume_byte_pos = (await aios.stat(partial_path)).st_size if (await aios.path.exists(partial_path)) else 0
            
            # This would call the new GetModelFile RPC to stream file chunks
            # Here we're just simulating the streaming behavior
            total_size = 0
            downloaded = resume_byte_pos
            
            async with aiofiles.open(partial_path, 'ab' if resume_byte_pos else 'wb') as f:
                async for chunk in peer.get_model_file(repo_id, revision, file_path, resume_byte_pos):
                    downloaded += await f.write(chunk.data)
                    total_size = chunk.total_size
                    on_progress(downloaded, total_size)
            
            # Rename from partial to final
            await aios.rename(partial_path, target_dir/file_path)
            return target_dir/file_path
            
        except Exception as e:
            if DEBUG >= 1:
                print(f"Error downloading file {file_path} from peer {peer.id()}: {e}")
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
        
        # If I'm the coordinator or we have no peers, download directly
        if self._coordinator_id == self.peers[0].id() or not self.peers:
            if DEBUG >= 2:
                print(f"I'm the coordinator or have no peers, downloading {repo_id} directly")
            return await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
            
        # Try to find a peer that has this model
        peer = await self.find_peer_with_model(repo_id)
        
        if peer:
            try:
                if DEBUG >= 1:
                    print(f"Downloading {repo_id} from peer {peer.id()}")
                    
                # Track that we're downloading this model
                self.downloading_models.add(repo_id)
                
                # Download from peer
                target_dir = await self.download_model_from_peer(
                    peer, 
                    shard, 
                    inference_engine_name, 
                    self.on_progress
                )
                
                # No longer downloading this model
                self.downloading_models.remove(repo_id)
                
                return target_dir
                
            except Exception as e:
                if DEBUG >= 1:
                    print(f"Failed to download {repo_id} from peer {peer.id()}, falling back to direct download: {e}")
                    traceback.print_exc()
                    
                # No longer downloading this model
                if repo_id in self.downloading_models:
                    self.downloading_models.remove(repo_id)
        
        # If we got here, either no peer had the model or download from peer failed
        # Fall back to direct download
        if DEBUG >= 1:
            print(f"Downloading {repo_id} directly from HuggingFace")
            
        return await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
    
    async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
        """Get the download status of all shards"""
        async for path, status in self.fallback_downloader.get_shard_download_status(inference_engine_name):
            yield path, status


def peer_shard_downloader(max_parallel_downloads: int = 8) -> ShardDownloader:
    """Create a new PeerShardDownloader with fallback to the standard downloader"""
    fallback = SingletonShardDownloader(CachedShardDownloader(NewShardDownloader(max_parallel_downloads)))
    return PeerShardDownloader(fallback, max_parallel_downloads)