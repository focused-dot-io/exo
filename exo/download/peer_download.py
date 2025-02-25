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
                                         max_wait_seconds=300, poll_interval_seconds=5.0):
        """Wait for the coordinator node to download a model
        
        Args:
            coordinator_peer: The peer handle for the coordinator node
            repo_id: The repo ID to wait for
            revision: The model revision to wait for (default: main)
            max_wait_seconds: Maximum time to wait in seconds (default: 300)
            poll_interval_seconds: How often to check if model is available (default: 5.0)
            
        Returns:
            bool: True if model was found on coordinator, False if timed out
        """
        print(f"[PEER DOWNLOAD] Waiting for coordinator {coordinator_peer.id()} to download model {repo_id}")
        print(f"[PEER DOWNLOAD] Will wait up to {max_wait_seconds} seconds with {poll_interval_seconds}s polling interval")
        
        # Print out debug info about gRPC connection
        try:
            is_connected = await coordinator_peer.is_connected()
            print(f"[PEER DOWNLOAD] Connection to coordinator: {'CONNECTED' if is_connected else 'NOT CONNECTED'}")
        except Exception as e:
            print(f"[PEER DOWNLOAD] Error checking connection to coordinator: {e}")
            if DEBUG >= 2:
                traceback.print_exc()
        
        start_time = time.time()
        attempts = 0
        downloading_seen = False
        
        # Check if the model is in the coordinator's downloading_models set
        try:
            # The coordinator might be in the process of downloading - check if it's already working on this model
            if repo_id in self.downloading_models:
                print(f"[PEER DOWNLOAD] Coordinator is currently downloading {repo_id}")
                downloading_seen = True
        except Exception:
            # If we can't access this information, just continue with the polling approach
            pass
        
        # Set the maximum timeout - we're more patient now!
        max_attempts = max_wait_seconds // poll_interval_seconds
        print(f"[PEER DOWNLOAD] Will make up to {max_attempts} attempts to check coordinator")
        
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
                    
                    # If the model is complete, we can download it
                    if has_model_response.is_complete:
                        print(f"[PEER DOWNLOAD] Coordinator has COMPLETE model - ready to download!")
                        return True
                    # If we've already waited a bit and the model is still incomplete, we'll still use it
                    elif attempts > 10 or time.time() - start_time > 60:
                        print(f"[PEER DOWNLOAD] Coordinator has partial model after waiting {time.time() - start_time:.1f}s, proceeding with download")
                        return True
                    else:
                        # Model is incomplete but we haven't waited long, keep polling
                        print(f"[PEER DOWNLOAD] Coordinator has incomplete model, continuing to wait...")
                        downloading_seen = True
                    
                # If we're debugging or periodically, log poll attempts
                if DEBUG >= 2 or attempts % 5 == 0:  # Log every 5 attempts
                    elapsed = time.time() - start_time
                    remaining = max_wait_seconds - elapsed
                    print(f"[PEER DOWNLOAD] Waiting for model {repo_id} on coordinator (elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s, attempt: {attempts})")
                    
            except Exception as e:
                print(f"[PEER DOWNLOAD] Error checking for model on coordinator: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
            
            # Wait before polling again
            await asyncio.sleep(poll_interval_seconds)
        
        print(f"[PEER DOWNLOAD] Timed out waiting for coordinator to download model {repo_id} after {max_wait_seconds} seconds")
        
        # Last attempt before giving up - if this fails, we'll try again later
        try:
            # One final check before giving up
            has_model_response = await coordinator_peer.has_model(repo_id, revision)
            if has_model_response.has_model:
                print(f"[PEER DOWNLOAD] Last-minute check: found model on coordinator!")
                return True
        except Exception:
            pass
            
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
        
        # First check specifically for the coordinator
        coordinator_peer = None
        if self._coordinator_id:
            for peer in self.peers:
                if peer.id() == self._coordinator_id:
                    coordinator_peer = peer
                    break
        
        # If coordinator is found, prioritize checking it first
        if coordinator_peer:
            try:
                if await coordinator_peer.is_connected():
                    print(f"[PEER DOWNLOAD] Checking coordinator: {coordinator_peer.id()}")
                    has_model_response = await coordinator_peer.has_model(repo_id, revision)
                    if has_model_response.has_model:
                        if has_model_response.is_complete:
                            print(f"[PEER DOWNLOAD] Coordinator {coordinator_peer.id()} has complete model {repo_id}")
                            return coordinator_peer
                        else:
                            print(f"[PEER DOWNLOAD] Coordinator {coordinator_peer.id()} has partial model {repo_id}")
                            best_peer = coordinator_peer
                            most_complete = False
                    else:
                        print(f"[PEER DOWNLOAD] Coordinator {coordinator_peer.id()} does not have model {repo_id}")
                else:
                    print(f"[PEER DOWNLOAD] Coordinator {coordinator_peer.id()} is not connected")
            except Exception as e:
                print(f"[PEER DOWNLOAD] Error checking if coordinator {coordinator_peer.id()} has model {repo_id}: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
        
        # If coordinator doesn't have it or there is no coordinator, check other peers
        if not self.peers:
            print(f"[PEER DOWNLOAD] No peers available to search for model {repo_id}")
            return None
            
        print(f"[PEER DOWNLOAD] Checking {len(self.peers)} peers for model {repo_id}")
        
        for i, peer in enumerate(self.peers):
            # Skip the coordinator if we already checked it
            if coordinator_peer and peer.id() == coordinator_peer.id():
                continue
                
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
        
        # Log on the sending side too (if we're the coordinator)
        try:
            if hasattr(peer, 'remote_id'):
                receiving_node_id = getattr(peer, 'remote_id', "unknown")
                print(f"[PEER DOWNLOAD] SENDING file {file_path} to node {receiving_node_id}")
        except:
            # If we can't get the remote ID, that's fine
            pass
            
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
            
            # Log for coordinator side
            try:
                if hasattr(peer, 'remote_id'):
                    receiving_node_id = getattr(peer, 'remote_id', "unknown")
                    print(f"[PEER DOWNLOAD] SENT file {file_path} ({downloaded/1024/1024:.2f} MB) to node {receiving_node_id}")
            except:
                pass
                
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
        """Download a full model from a peer
        
        This method is called by the non-coordinator nodes to download from the coordinator,
        or by any node to download from a peer that has the model.
        """
        repo_id = get_repo(shard.model_id, inference_engine_name)
        revision = "main"
        target_dir = await ensure_downloads_dir()/repo_id.replace("/", "--")
        await aios.makedirs(target_dir, exist_ok=True)
        
        # Get list of files from peer
        # Use '*' to ensure we get ALL files, not just tensors
        allow_patterns = ["*"]  # This will override the specific patterns and get everything
        
        print(f"[PEER DOWNLOAD] Getting complete file list for {repo_id} from peer {peer.id()}")
        
        # Log on the sending side too (if we're sending model to another node)
        try:
            if hasattr(peer, 'remote_id'):
                print(f"[PEER DOWNLOAD] Preparing to SEND model {repo_id} to node {peer.remote_id}")
        except:
            # If we can't get the remote ID, that's fine
            pass
            
        # Check if peer is connected before trying to get file list
        try:
            is_connected = await peer.is_connected()
            if not is_connected:
                print(f"[PEER DOWNLOAD] WARNING: Peer {peer.id()} is not connected! Attempting to connect...")
                try:
                    await peer.connect()
                    print(f"[PEER DOWNLOAD] Successfully connected to peer {peer.id()}")
                except Exception as e:
                    print(f"[PEER DOWNLOAD] Failed to connect to peer {peer.id()}: {e}")
                    if DEBUG >= 2:
                        traceback.print_exc()
        except Exception as e:
            print(f"[PEER DOWNLOAD] Error checking connection status: {e}")
            
        # Now get the file list
        file_list_response = await peer.get_model_file_list(repo_id, revision, allow_patterns)
        
        if not file_list_response.files:
            # If this is the coordinator, continue waiting instead of raising error
            if peer.id() == self._coordinator_id:
                print(f"[PEER DOWNLOAD] Coordinator {peer.id()} does not have model files yet for {repo_id}")
                print(f"[PEER DOWNLOAD] Coordinator is likely still starting the download")
                print(f"[PEER DOWNLOAD] Will wait for coordinator to download model...")
                
                # Wait for coordinator to download model
                wait_success = await self._wait_for_model_on_coordinator(
                    peer, repo_id, max_wait_seconds=300, poll_interval_seconds=5.0
                )
                
                if wait_success:
                    print(f"[PEER DOWNLOAD] Coordinator now has model, trying to get file list again")
                    # Try again after waiting
                    file_list_response = await peer.get_model_file_list(repo_id, revision, allow_patterns)
                    
                    if not file_list_response.files:
                        raise Exception(f"Coordinator {peer.id()} returned empty file list for {repo_id} even after waiting")
                else:
                    raise Exception(f"Timed out waiting for coordinator to download {repo_id}")
            else:
                # For non-coordinator peers, just raise the error
                raise Exception(f"Peer {peer.id()} returned empty file list for {repo_id}")
            
        # Convert the file list to the format our progress tracking expects
        file_list = [{"path": f.path, "size": f.size} for f in file_list_response.files]
        
        # Debug log to show what files we're going to download
        if DEBUG >= 2:
            print(f"[PEER DOWNLOAD] Found {len(file_list)} files to download:")
            for file in file_list:
                print(f"[PEER DOWNLOAD]   - {file['path']} ({file['size']/1024/1024:.2f} MB)")
        else:
            total_size_mb = sum(f['size'] for f in file_list)/1024/1024
            print(f"[PEER DOWNLOAD] Found {len(file_list)} files to download (total size: {total_size_mb:.2f} MB)")
            
            # Create a status event to show in the UI
            from exo.download.download_progress import RepoProgressEvent
            download_start_event = RepoProgressEvent(
                repo_id=repo_id,
                revision=revision,
                file_progress={},
                status=f"Peer download starting: {len(file_list)} files, {total_size_mb:.2f} MB",
                progress=0.0,
                eta=timedelta(seconds=0),
                speed=0,
                start_time=time.time()
            )
            on_progress.trigger_all(shard, download_start_event)
        
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
            missing_files = []
            for file in filtered_file_list:
                if not await aios.path.exists(target_dir/file["path"]):
                    all_files_exist = False
                    missing_files.append(file["path"])
            
            # Extra check for weight files (critical for inference)
            has_weight_files = False
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if (file.endswith(".safetensors") or file.endswith(".bin") or 
                        file.endswith(".pt") or file.endswith(".gguf")):
                        has_weight_files = True
                        break
                if has_weight_files:
                    break
            
            # Only return if we have all files AND weight files
            if all_files_exist and has_weight_files:
                print(f"[PEER DOWNLOAD] Model {repo_id} is already downloaded and complete")
                return target_dir
            elif not has_weight_files:
                print(f"[PEER DOWNLOAD] WARNING: Model {repo_id} is missing weight files (.safetensors)")
                # Continue to get proper model from coordinator
            elif not all_files_exist:
                print(f"[PEER DOWNLOAD] Model {repo_id} is partially downloaded (missing {len(missing_files)} files)")
                # Continue to get complete model
        
        # If I have no peers, download directly
        if not self.peers:
            print(f"[PEER DOWNLOAD] No peers available, downloading {repo_id} directly")
            return await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
            
        # Check if I'm the coordinator - ensure we calculate this reliably
        am_i_coordinator = False
        my_id = None
        
        if self._coordinator_id:
            # If we have peers with IDs, we can compare to determine coordinator status
            if hasattr(self, 'peers') and self.peers:
                # Get my node ID from either my own peer object or the first peer's method
                for peer in self.peers:
                    my_id = peer.id()
                    # Don't break after getting an ID - keep looking until we find our own
                    if hasattr(peer, 'own_id') and peer.own_id is not None:
                        my_id = peer.own_id
                        break
                
                if my_id:
                    am_i_coordinator = (self._coordinator_id == my_id)
                    print(f"[PEER DOWNLOAD] Coordinator check: My ID={my_id}, Coordinator ID={self._coordinator_id}, am_i_coordinator={am_i_coordinator}")
                else:
                    print(f"[PEER DOWNLOAD] Unable to determine my node ID from peers")
            else:
                print(f"[PEER DOWNLOAD] Unable to determine coordinator status due to missing peer ID")
                
        # If I'm definitely the coordinator, download directly
        if am_i_coordinator:
            print(f"[PEER DOWNLOAD] I am the coordinator node ({self._coordinator_id}), downloading {repo_id} directly")
            return await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
            
        # Otherwise I should try to download from peers
        my_id = self.peers[0].id() if self.peers else "unknown"
        print(f"[PEER DOWNLOAD] I am NOT the coordinator. My ID: {my_id}, Coordinator: {self._coordinator_id}")
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
            coordinator_has_complete_model = has_model_response.is_complete if has_model_response.has_model else False
            
            if not coordinator_has_model:
                # Wait for coordinator to download the model
                print(f"[PEER DOWNLOAD] Coordinator does not have model {repo_id} yet, waiting for it to download...")
                
                # Use our wait method to poll for coordinator to complete download
                wait_success = await self._wait_for_model_on_coordinator(coordinator_peer, repo_id)
                
                if wait_success:
                    print(f"[PEER DOWNLOAD] Coordinator now has model {repo_id}, will download from coordinator")
                    
                    # Create a status event to update the UI that we're waiting
                    try:
                        from exo.download.download_progress import RepoProgressEvent
                        status_event = RepoProgressEvent(
                            repo_id=repo_id,
                            revision="main",
                            file_progress={},
                            status=f"Preparing to download {repo_id} from coordinator",
                            progress=0.0,
                            eta=timedelta(seconds=0),
                            speed=0,
                            start_time=time.time()
                        )
                        # Try to trigger the event if we have an on_progress callback
                        if hasattr(self, 'on_progress'):
                            self.on_progress.trigger_all(shard, status_event)
                    except Exception:
                        # Don't fail if there's an error updating status
                        pass
            elif not coordinator_has_complete_model:
                # Coordinator has model but it's incomplete - wait for it to complete
                print(f"[PEER DOWNLOAD] Coordinator has incomplete model {repo_id}, waiting for it to finish downloading...")
                
                # Create a status event to update the UI that we're waiting for coordinator
                try:
                    from exo.download.download_progress import RepoProgressEvent
                    waiting_event = RepoProgressEvent(
                        repo_id=repo_id,
                        revision="main",
                        file_progress={},
                        status=f"Waiting for coordinator to complete downloading {repo_id}",
                        progress=0.0,
                        eta=timedelta(seconds=0),
                        speed=0,
                        start_time=time.time()
                    )
                    # Try to trigger the event if we have an on_progress callback
                    if hasattr(self, 'on_progress'):
                        self.on_progress.trigger_all(shard, waiting_event)
                except Exception:
                    # Don't fail if there's an error updating status
                    pass
                
                # Use our wait method to poll for coordinator to complete download
                wait_success = await self._wait_for_model_on_coordinator(coordinator_peer, repo_id)
                
                if wait_success:
                    print(f"[PEER DOWNLOAD] Coordinator now has model {repo_id} (complete or usable), will download from coordinator")
        
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
                
                # Special case for coordinator - if coordinator is still in early stages of downloading,
                # we might get a "has_model=True" but empty file list. Handle this specially.
                if peer.id() == self._coordinator_id:
                    print(f"[PEER DOWNLOAD] Downloading from coordinator {peer.id()}")
                    
                    # First check if this is just the start of the download
                    file_list_response = await peer.get_model_file_list(repo_id, revision, ["*"])
                    if not file_list_response.files:
                        print(f"[PEER DOWNLOAD] Coordinator {peer.id()} has model but no files yet - waiting for download to progress")
                        # Wait for coordinator to make progress
                        wait_success = await self._wait_for_model_on_coordinator(
                            peer, repo_id, max_wait_seconds=300, poll_interval_seconds=5.0
                        )
                
                target_dir = await self.download_model_from_peer(
                    peer, 
                    shard, 
                    inference_engine_name, 
                    self.on_progress
                )
                download_time = time.time() - download_start
                
                # No longer downloading this model
                if repo_id in self.downloading_models:
                    self.downloading_models.remove(repo_id)
                
                print(f"[PEER DOWNLOAD] Successfully downloaded {repo_id} from peer {peer.id()} in {download_time:.2f} seconds")
                return target_dir
                
            except Exception as e:
                print(f"[PEER DOWNLOAD] Failed to download {repo_id} from peer {peer.id()}")
                print(f"[PEER DOWNLOAD] Error: {e}")
                if DEBUG >= 2:
                    traceback.print_exc()
                
                # If this was the coordinator and we failed, we should wait and retry
                # instead of falling back to direct download (for non-coordinator nodes)
                if peer.id() == self._coordinator_id and self._coordinator_id != self.peers[0].id():
                    am_i_coordinator = False
                    if self._coordinator_id:
                        my_id = self.peers[0].id() if self.peers else "unknown"
                        am_i_coordinator = (self._coordinator_id == my_id)
                    
                    if not am_i_coordinator:
                        print(f"[PEER DOWNLOAD] Will wait and retry downloading from coordinator")
                        # Continue to wait-and-retry logic below instead of falling back
                    
                # No longer downloading this model
                if repo_id in self.downloading_models:
                    self.downloading_models.remove(repo_id)
        
        # If we got here, either no peer had the model or download from peer failed
        # Check if I'm the coordinator - only the coordinator should fall back to direct download
        # Re-check our coordinator status to be certain
        am_i_coordinator = False
        my_id = None
        
        if self._coordinator_id:
            if hasattr(self, 'peers') and self.peers:
                # Try to find our own ID with higher confidence
                for peer in self.peers:
                    my_id = peer.id()
                    if hasattr(peer, 'own_id') and peer.own_id is not None:
                        my_id = peer.own_id
                        break
                
                if my_id:
                    am_i_coordinator = (self._coordinator_id == my_id)
                    print(f"[PEER DOWNLOAD] Double-checking coordinator status: My ID={my_id}, Coordinator ID={self._coordinator_id}, am_i_coordinator={am_i_coordinator}")
            
        if am_i_coordinator:
            # ONLY the coordinator should fall back to direct download
            print(f"[PEER DOWNLOAD] No peers have {repo_id} or peer download failed")
            print(f"[PEER DOWNLOAD] I am the coordinator, falling back to direct download from HuggingFace")
                
            direct_download_start = time.time()
            result = await self.fallback_downloader.ensure_shard(shard, inference_engine_name)
            direct_download_time = time.time() - direct_download_start
            
            print(f"[PEER DOWNLOAD] Direct download of {repo_id} completed in {direct_download_time:.2f} seconds")
            return result
        else:
            # Non-coordinator nodes should NEVER fall back to direct download
            print(f"[PEER DOWNLOAD] No peers have {repo_id} yet, but I am not the coordinator")
            print(f"[PEER DOWNLOAD] Will continue waiting for coordinator ({self._coordinator_id}) to download model")
            
            # Create a status event to update the UI that we're waiting for the coordinator
            try:
                from exo.download.download_progress import RepoProgressEvent
                waiting_event = RepoProgressEvent(
                    repo_id=repo_id,
                    revision="main",
                    file_progress={},
                    status=f"Waiting for coordinator to download model {repo_id}...",
                    progress=0.0,
                    eta=timedelta(seconds=0),
                    speed=0,
                    start_time=time.time()
                )
                # Try to trigger the event if we have an on_progress callback
                if hasattr(self, 'on_progress'):
                    self.on_progress.trigger_all(shard, waiting_event)
            except Exception:
                # Don't fail if there's an error updating status
                pass
            
            # Keep waiting for the coordinator, with retries if needed
            max_retries = 5
            for retry in range(max_retries):
                # First find the coordinator peer (it might have reconnected)
                coordinator_peer = None
                for peer in self.peers:
                    if peer.id() == self._coordinator_id:
                        coordinator_peer = peer
                        break
                        
                if coordinator_peer:
                    print(f"[PEER DOWNLOAD] Waiting up to 300 seconds for coordinator to download model (attempt {retry+1}/{max_retries})...")
                    
                    # Update the UI with the retry attempt
                    try:
                        from exo.download.download_progress import RepoProgressEvent
                        retry_event = RepoProgressEvent(
                            repo_id=repo_id,
                            revision="main",
                            file_progress={},
                            status=f"Waiting for coordinator (attempt {retry+1}/{max_retries})...",
                            progress=float(retry) / max_retries,  # Show some progress in the UI
                            eta=timedelta(seconds=300),
                            speed=0,
                            start_time=time.time()
                        )
                        if hasattr(self, 'on_progress'):
                            self.on_progress.trigger_all(shard, retry_event)
                    except Exception:
                        pass
                        
                    wait_success = await self._wait_for_model_on_coordinator(
                        coordinator_peer, repo_id, max_wait_seconds=300, poll_interval_seconds=5.0
                    )
                    
                    if wait_success:
                        print(f"[PEER DOWNLOAD] Coordinator now has model {repo_id}, downloading from coordinator")
                        try:
                            result = await self.download_model_from_peer(
                                coordinator_peer, shard, inference_engine_name, self.on_progress
                            )
                            print(f"[PEER DOWNLOAD] Successfully downloaded model from coordinator")
                            return result
                        except Exception as e:
                            print(f"[PEER DOWNLOAD] Error downloading from coordinator: {e}")
                            print(f"[PEER DOWNLOAD] Will retry waiting for complete model")
                            # Continue to next retry
                else:
                    print(f"[PEER DOWNLOAD] Coordinator peer {self._coordinator_id} not found (attempt {retry+1}/{max_retries})")
                    print(f"[PEER DOWNLOAD] Waiting 30 seconds for peers to reconnect...")
                    await asyncio.sleep(30)  # Wait for peers to potentially reconnect
                    
                    # The PeerShardDownloader can't update its own peers list here
                    # This is a limitation - it depends on someone else setting its peers through set_peers()
                    # This update would need to be handled at the Node level, which calls set_peers() periodically
                    print(f"[PEER DOWNLOAD] Cannot automatically update peer list - waiting for external peer updates")
                            
            # If we get here, we've tried multiple times and still can't download the model
            print(f"[PEER DOWNLOAD] ERROR: Could not download model after {max_retries} attempts")
            print(f"[PEER DOWNLOAD] Please check that the coordinator is running and downloading the model")
            print(f"[PEER DOWNLOAD] Suspending inference until model is available - will not use empty directory")
            
            # Update UI to show we're in continuous wait mode
            try:
                from exo.download.download_progress import RepoProgressEvent
                continuous_wait_event = RepoProgressEvent(
                    repo_id=repo_id,
                    revision="main",
                    file_progress={},
                    status=f"Entering continuous wait mode for model {repo_id}...",
                    progress=0.1,  # Show a small amount of progress
                    eta=timedelta(seconds=0),  # Unknown ETA
                    speed=0,
                    start_time=time.time()
                )
                if hasattr(self, 'on_progress'):
                    self.on_progress.trigger_all(shard, continuous_wait_event)
            except Exception:
                pass
            
            # Keep retrying forever at this point, but with longer intervals
            print(f"[PEER DOWNLOAD] Entering continuous wait mode - will check every 60 seconds")
            check_count = 0
            while True:
                check_count += 1
                await asyncio.sleep(60)
                
                # Update status every 5 minutes in the UI
                if check_count % 5 == 0:
                    try:
                        wait_time_mins = check_count
                        from exo.download.download_progress import RepoProgressEvent
                        long_wait_event = RepoProgressEvent(
                            repo_id=repo_id,
                            revision="main",
                            file_progress={},
                            status=f"Still waiting for coordinator ({wait_time_mins} minutes)...",
                            progress=min(0.2 + (check_count / 100.0), 0.9),  # Slowly increase progress but never complete
                            eta=timedelta(seconds=0),
                            speed=0,
                            start_time=time.time()
                        )
                        if hasattr(self, 'on_progress'):
                            self.on_progress.trigger_all(shard, long_wait_event)
                    except Exception:
                        pass
                        
                print(f"[PEER DOWNLOAD] Still waiting for coordinator to download model... (check {check_count})")
                
                # Find the coordinator peer again
                coordinator_peer = None
                for peer in self.peers:
                    if peer.id() == self._coordinator_id:
                        coordinator_peer = peer
                        break
                        
                if coordinator_peer:
                    wait_success = await self._wait_for_model_on_coordinator(
                        coordinator_peer, repo_id, max_wait_seconds=10, poll_interval_seconds=2.0
                    )
                    
                    if wait_success:
                        print(f"[PEER DOWNLOAD] Finally found model on coordinator, downloading now")
                        return await self.download_model_from_peer(
                            coordinator_peer, shard, inference_engine_name, self.on_progress
                        )
    
    async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
        """Get the download status of all shards"""
        async for path, status in self.fallback_downloader.get_shard_download_status(inference_engine_name):
            yield path, status


def peer_shard_downloader(max_parallel_downloads: int = 8) -> ShardDownloader:
    """Create a new PeerShardDownloader with fallback to the standard downloader"""
    fallback = SingletonShardDownloader(CachedShardDownloader(NewShardDownloader(max_parallel_downloads)))
    return PeerShardDownloader(fallback, max_parallel_downloads)