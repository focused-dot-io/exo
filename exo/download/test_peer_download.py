import pytest
import asyncio
import os
import json
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from typing import List, Dict, Optional, AsyncIterator

from exo.inference.shard import Shard
from exo.networking.peer_handle import PeerHandle, HasModelResponse, ModelFileListResponse, ModelFileInfo, FileChunk
from exo.download.peer_download import PeerShardDownloader
from exo.download.shard_download import ShardDownloader, NoopShardDownloader


class MockPeerHandle(PeerHandle):
    def __init__(self, _id: str, has_model: bool = False, is_complete: bool = False, files: List[Dict] = None):
        self._id = _id
        self._has_model = has_model
        self._is_complete = is_complete
        self._files = files or []
        self._connected = True
        
        # Track method calls for validation
        self.has_model_calls = []
        self.get_model_file_list_calls = []
        self.get_model_file_calls = []
        
    def id(self) -> str:
        return self._id
        
    def addr(self) -> str:
        return f"mock-addr-{self._id}"
        
    def description(self) -> str:
        return f"Mock peer {self._id}"
        
    def device_capabilities(self):
        return None
        
    async def connect(self) -> None:
        self._connected = True
        
    async def is_connected(self) -> bool:
        return self._connected
        
    async def disconnect(self) -> None:
        self._connected = False
        
    async def health_check(self) -> bool:
        return self._connected
        
    async def send_prompt(self, shard, prompt, request_id=None):
        return None
        
    async def send_tensor(self, shard, tensor, request_id=None):
        return None
        
    async def send_result(self, request_id, result, is_finished):
        return None
        
    async def collect_topology(self, visited, max_depth):
        return None
        
    async def has_model(self, repo_id: str, revision: str = "main") -> HasModelResponse:
        self.has_model_calls.append((repo_id, revision))
        return HasModelResponse(
            has_model=self._has_model,
            is_complete=self._is_complete,
            available_files=[f["path"] for f in self._files]
        )
        
    async def get_model_file_list(self, repo_id: str, revision: str = "main", allow_patterns: List[str] = None) -> ModelFileListResponse:
        self.get_model_file_list_calls.append((repo_id, revision, allow_patterns))
        files = [
            ModelFileInfo(path=f["path"], size=f["size"], hash=f.get("hash", ""))
            for f in self._files
        ]
        return ModelFileListResponse(files=files)
        
    async def get_model_file(self, repo_id: str, revision: str, file_path: str, offset: int = 0) -> AsyncIterator[FileChunk]:
        self.get_model_file_calls.append((repo_id, revision, file_path, offset))
        
        # Find the matching file
        file_info = next((f for f in self._files if f["path"] == file_path), None)
        if not file_info:
            return
            
        # Mock file content
        content = b"x" * file_info["size"]
        chunk_size = 1024 * 1024  # 1MB chunks
        
        total_size = file_info["size"]
        current_offset = offset
        
        while current_offset < total_size:
            chunk_end = min(current_offset + chunk_size, total_size)
            chunk_size_actual = chunk_end - current_offset
            
            yield FileChunk(
                data=content[:chunk_size_actual],
                offset=current_offset,
                total_size=total_size
            )
            
            current_offset = chunk_end
            if current_offset >= total_size:
                break


class MockFallbackDownloader(ShardDownloader):
    def __init__(self):
        self._on_progress = AsyncMock()
        self.ensure_shard_calls = []
        
    @property
    def on_progress(self):
        return self._on_progress
        
    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        self.ensure_shard_calls.append((shard, inference_engine_name))
        return Path(f"/mock/downloads/{shard.model_id}")
        
    async def get_shard_download_status(self, inference_engine_name: str):
        if False:  # Just to make it an async generator
            yield


@pytest.mark.asyncio
async def test_peer_downloader_initialization():
    """Test that the peer downloader initializes correctly"""
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback, max_parallel_downloads=4)
    
    assert downloader.fallback_downloader == fallback
    assert downloader.max_parallel_downloads == 4
    assert downloader.peers == []
    assert downloader._coordinator_id is None
    assert downloader.downloading_models == set()


@pytest.mark.asyncio
async def test_set_peers_and_coordinator():
    """Test that peers and coordinator can be set properly"""
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback)
    
    # Create mock peers
    peers = [MockPeerHandle(f"peer{i}") for i in range(3)]
    
    # Set peers
    downloader.set_peers(peers)
    assert len(downloader.peers) == 3
    
    # Set coordinator
    downloader.set_coordinator_id("coordinator-123")
    assert downloader._coordinator_id == "coordinator-123"


@pytest.mark.asyncio
async def test_find_peer_with_model():
    """Test that the downloader can find a peer with a model"""
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback)
    
    # Create mock peers with different model availability
    peer1 = MockPeerHandle("peer1", has_model=False)
    peer2 = MockPeerHandle("peer2", has_model=True, is_complete=False)
    peer3 = MockPeerHandle("peer3", has_model=True, is_complete=True)
    
    # Set all peers
    downloader.set_peers([peer1, peer2, peer3])
    
    # Should find peer3 first because it has a complete model
    best_peer = await downloader.find_peer_with_model("test-model-repo")
    assert best_peer.id() == "peer3"
    
    # Verify the has_model method was called on all peers
    assert len(peer1.has_model_calls) == 1
    assert len(peer2.has_model_calls) == 1
    assert len(peer3.has_model_calls) == 1


@pytest.mark.asyncio
async def test_find_peer_with_model_incomplete():
    """Test that the downloader can find a peer with an incomplete model if no complete ones exist"""
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback)
    
    # Create mock peers with different model availability
    peer1 = MockPeerHandle("peer1", has_model=False)
    peer2 = MockPeerHandle("peer2", has_model=True, is_complete=False)
    
    # Set all peers
    downloader.set_peers([peer1, peer2])
    
    # Should find peer2 because it has a model, even if incomplete
    best_peer = await downloader.find_peer_with_model("test-model-repo")
    assert best_peer.id() == "peer2"


@pytest.mark.asyncio
@patch('exo.download.peer_download.ensure_downloads_dir')
@patch('exo.download.peer_download.aios.makedirs')
@patch('exo.download.peer_download.aios.path.exists')
async def test_download_file_from_peer(mock_exists, mock_makedirs, mock_ensure_downloads_dir, tmp_path):
    """Test downloading a single file from a peer"""
    mock_ensure_downloads_dir.return_value = Path("/mock/downloads")
    mock_exists.return_value = False
    
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback)
    
    # Create a mock peer with a file
    mock_file = {"path": "model.safetensors", "size": 1024 * 1024}  # 1MB file
    peer = MockPeerHandle("peer1", has_model=True, is_complete=True, files=[mock_file])
    
    # Set up the aiofiles.open mock
    mock_file_handle = AsyncMock()
    mock_file_handle.__aenter__.return_value.write = AsyncMock(return_value=1024 * 1024)
    
    with patch('aiofiles.open', return_value=mock_file_handle):
        with patch('exo.download.peer_download.aios.rename', new_callable=AsyncMock) as mock_rename:
            # Download the file
            progress_callback = MagicMock()
            result = await downloader.download_file_from_peer(
                peer,
                "test-repo",
                "main",
                "model.safetensors",
                Path("/mock/downloads/test-repo"),
                progress_callback
            )
            
            # Verify the file was downloaded
            assert result == Path("/mock/downloads/test-repo/model.safetensors")
            
            # Verify progress callback was called
            assert progress_callback.called
            
            # Verify rename was called (from .partial to final file)
            assert mock_rename.called


@pytest.mark.asyncio
@patch('exo.download.peer_download.aios.path.exists')
@patch('exo.download.peer_download.get_repo')
@patch('exo.download.peer_download.resolve_allow_patterns')
async def test_ensure_shard_with_existing_model(mock_resolve_patterns, mock_get_repo, mock_exists):
    """Test that ensure_shard returns an existing model if it's already downloaded"""
    mock_get_repo.return_value = "test-repo/model"
    mock_exists.return_value = True
    mock_resolve_patterns.return_value = ["*"]
    
    with patch('exo.download.peer_download.fetch_file_list_with_cache', new_callable=AsyncMock) as mock_fetch_list:
        mock_fetch_list.return_value = [{"path": "model.safetensors", "size": 1024}]
        
        # Use the local filter_repo_objects implementation now
        with patch.object(PeerShardDownloader, 'filter_repo_objects', return_value=[{"path": "model.safetensors", "size": 1024}]):
            
            with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
                mock_ensure_dir.return_value = Path("/mock/downloads")
                
                fallback = MockFallbackDownloader()
                downloader = PeerShardDownloader(fallback)
                shard = Shard("test-model", 0, 1, 2)
                
                result = await downloader.ensure_shard(shard, "test-engine")
                
                # Verify the model was found locally
                assert result == Path("/mock/downloads/test-repo--model")
                
                # Verify the fallback downloader wasn't called
                assert len(fallback.ensure_shard_calls) == 0


@pytest.mark.asyncio
@patch('exo.download.peer_download.aios.path.exists')
@patch('exo.download.peer_download.get_repo')
async def test_ensure_shard_as_coordinator(mock_get_repo, mock_exists):
    """Test that coordinator downloads directly from HuggingFace"""
    mock_get_repo.return_value = "test-repo/model"
    mock_exists.return_value = False
    
    with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
        mock_ensure_dir.return_value = Path("/mock/downloads")
        
        fallback = MockFallbackDownloader()
        downloader = PeerShardDownloader(fallback)
        
        # Set up peers and make this node the coordinator
        peer1 = MockPeerHandle("peer1")
        downloader.set_peers([peer1])
        downloader.set_coordinator_id("peer1")
        
        shard = Shard("test-model", 0, 1, 2)
        
        result = await downloader.ensure_shard(shard, "test-engine")
        
        # Verify the fallback downloader was called
        assert len(fallback.ensure_shard_calls) == 1
        called_shard, called_engine = fallback.ensure_shard_calls[0]
        assert called_shard.model_id == "test-model"
        assert called_engine == "test-engine"


@pytest.mark.asyncio
@patch('exo.download.peer_download.aios.path.exists')
@patch('exo.download.peer_download.get_repo')
async def test_ensure_shard_from_peer(mock_get_repo, mock_exists):
    """Test downloading a model from a peer that has it"""
    mock_get_repo.return_value = "test-repo/model"
    mock_exists.return_value = False
    
    with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
        mock_ensure_dir.return_value = Path("/mock/downloads")
        
        fallback = MockFallbackDownloader()
        downloader = PeerShardDownloader(fallback)
        
        # Create peers, one with the model
        peer1 = MockPeerHandle("peer1", has_model=False)
        peer2 = MockPeerHandle("peer2", has_model=True, is_complete=True, files=[
            {"path": "model.safetensors", "size": 1024}
        ])
        
        # Set peers and make a different node the coordinator
        downloader.set_peers([peer1, peer2])
        downloader.set_coordinator_id("coordinator")
        
        # Mock the download_model_from_peer method
        download_mock = AsyncMock(return_value=Path("/mock/downloads/test-repo--model"))
        downloader.download_model_from_peer = download_mock
        
        shard = Shard("test-model", 0, 1, 2)
        
        result = await downloader.ensure_shard(shard, "test-engine")
        
        # Verify we tried to find a peer with the model
        assert len(peer1.has_model_calls) == 1
        assert len(peer2.has_model_calls) == 1
        
        # Verify download_model_from_peer was called with the right peer
        download_mock.assert_called_once()
        called_peer, called_shard, called_engine, _ = download_mock.call_args[0]
        assert called_peer.id() == "peer2"
        assert called_shard.model_id == "test-model"
        assert called_engine == "test-engine"
        
        # Verify fallback wasn't used
        assert len(fallback.ensure_shard_calls) == 0


@pytest.mark.asyncio
@patch('exo.download.peer_download.aios.path.exists')
@patch('exo.download.peer_download.get_repo')
async def test_ensure_shard_fallback_when_peer_download_fails(mock_get_repo, mock_exists):
    """Test falling back to direct download when peer download fails"""
    mock_get_repo.return_value = "test-repo/model"
    mock_exists.return_value = False
    
    with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
        mock_ensure_dir.return_value = Path("/mock/downloads")
        
        fallback = MockFallbackDownloader()
        downloader = PeerShardDownloader(fallback)
        
        # Create a peer with the model
        peer = MockPeerHandle("peer1", has_model=True, is_complete=True)
        downloader.set_peers([peer])
        downloader.set_coordinator_id("coordinator")
        
        # Make the download_model_from_peer method fail
        download_mock = AsyncMock(side_effect=Exception("Download failed"))
        downloader.download_model_from_peer = download_mock
        
        shard = Shard("test-model", 0, 1, 2)
        
        result = await downloader.ensure_shard(shard, "test-engine")
        
        # Verify download_model_from_peer was attempted
        download_mock.assert_called_once()
        
        # Verify fallback was used
        assert len(fallback.ensure_shard_calls) == 1
        called_shard, called_engine = fallback.ensure_shard_calls[0]
        assert called_shard.model_id == "test-model"
        assert called_engine == "test-engine"


@pytest.mark.asyncio
async def test_peer_downloader_detailed_logging(capsys):
    """Test that the peer downloader logs with [PEER DOWNLOAD] tag"""
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback)
    
    # Create mock peers with different model availability
    peer1 = MockPeerHandle("peer1", has_model=False)
    peer2 = MockPeerHandle("peer2", has_model=True, is_complete=True)
    
    # Set all peers
    downloader.set_peers([peer1, peer2])
    downloader.set_coordinator_id("coordinator-node")
    
    # Find peer with model
    best_peer = await downloader.find_peer_with_model("test-model-repo")
    
    # Capture the output and verify the log messages contain [PEER DOWNLOAD]
    captured = capsys.readouterr()
    assert "[PEER DOWNLOAD]" in captured.out
    assert "Searching for peer with model" in captured.out
    assert "Found peer" in captured.out

@pytest.mark.asyncio
async def test_coordinator_based_download_decision():
    """Test that the download path depends on coordinator status"""
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback)
    
    # Create mock shard and peer
    shard = Shard("test-model", 0, 1, 2)
    peer = MockPeerHandle("peer1", has_model=True, is_complete=True)
    
    with patch('exo.download.peer_download.get_repo', return_value="test-repo/model"):
        with patch('exo.download.peer_download.aios.path.exists', return_value=False):
            with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
                mock_ensure_dir.return_value = Path("/mock/downloads")
                
                # Case 1: I am not the coordinator and there are peers
                downloader.set_peers([peer])
                downloader.set_coordinator_id("coordinator-node")
                
                # Mock the find_peer_with_model method
                downloader.find_peer_with_model = AsyncMock(return_value=None)
                
                # Should try to find peer first
                await downloader.ensure_shard(shard, "test-engine")
                downloader.find_peer_with_model.assert_called_once()
                
                # Case 2: I am the coordinator
                downloader.find_peer_with_model.reset_mock()
                downloader.set_coordinator_id(peer.id())
                
                # Should download directly without trying to find peer
                await downloader.ensure_shard(shard, "test-engine")
                downloader.find_peer_with_model.assert_not_called()

@pytest.mark.asyncio
async def test_download_performance_metrics(capsys):
    """Test that the download methods report performance metrics"""
    fallback = MockFallbackDownloader()
    downloader = PeerShardDownloader(fallback)
    
    # Create a mock peer with a file
    mock_file = {"path": "model.safetensors", "size": 1024 * 1024}  # 1MB file
    peer = MockPeerHandle("peer1", has_model=True, is_complete=True, files=[mock_file])
    
    # Set up the aiofiles.open mock
    mock_file_handle = AsyncMock()
    mock_file_handle.__aenter__.return_value.write = AsyncMock(return_value=1024 * 1024)
    
    with patch('aiofiles.open', return_value=mock_file_handle):
        with patch('exo.download.peer_download.aios.rename', new_callable=AsyncMock) as mock_rename:
            with patch('exo.download.peer_download.aios.makedirs', new_callable=AsyncMock):
                with patch('exo.download.peer_download.aios.path.exists', return_value=False):
                    # Download the file
                    progress_callback = MagicMock()
                    result = await downloader.download_file_from_peer(
                        peer,
                        "test-repo",
                        "main",
                        "model.safetensors",
                        Path("/mock/downloads/test-repo"),
                        progress_callback
                    )
                    
                    # Check that performance metrics were logged
                    captured = capsys.readouterr()
                    assert "[PEER DOWNLOAD]" in captured.out
                    assert "Successfully downloaded" in captured.out
                    assert "MB/s" in captured.out  # Speed metrics
                    assert "s (" in captured.out  # Time metrics with seconds

if __name__ == "__main__":
    pytest.main(["-xvs", "test_peer_download.py"])