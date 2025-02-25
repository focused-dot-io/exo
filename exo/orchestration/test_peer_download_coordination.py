import pytest
import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from exo.inference.shard import Shard
from exo.networking.peer_handle import PeerHandle, HasModelResponse, ModelFileListResponse, ModelFileInfo, FileChunk
from exo.download.peer_download import PeerShardDownloader
from exo.download.shard_download import ShardDownloader
from exo.orchestration.node import Node
from exo.networking.server import Server
from exo.networking.discovery import Discovery
from exo.inference.inference_engine import InferenceEngine
from exo.topology.partitioning_strategy import PartitioningStrategy
from exo.topology.device_capabilities import device_capabilities
from exo.helpers import DEBUG

# Import MockPeerHandle from the unit tests
from exo.download.test_peer_download import MockPeerHandle, MockFallbackDownloader


# Create a subclass of Node that doesn't create background tasks for testing
# Use a name that doesn't start with 'Test' to avoid pytest trying to collect it
class NonBackgroundTaskNode(Node):
    """A subclass of Node that doesn't create background tasks for testing"""
    
    async def start(self, wait_for_peers: int = 0) -> None:
        """Override start to not create background tasks"""
        self.device_capabilities = await device_capabilities()
        await self.server.start()
        await self.discovery.start()
        await self.update_peers(wait_for_peers)
        await self.collect_topology(set())
        if DEBUG >= 2: print(f"Collected topology: {self.topology}")
        
        # Skip periodic_topology_collection to avoid test warnings
        
        # Setup download coordinator - first node discovered becomes the coordinator
        await self.setup_peer_download_coordinator()

class MockPeerShardDownloader(PeerShardDownloader):
    """A mock peer shard downloader for testing"""
    def __init__(self, fallback_downloader):
        super().__init__(fallback_downloader)
        self.download_model_from_peer_calls = []

    async def download_model_from_peer(self, peer, shard, inference_engine_name, on_progress):
        self.download_model_from_peer_calls.append((peer.id(), shard, inference_engine_name))
        return Path(f"/mock/downloads/{shard.model_name}")


class MockInferenceEngine(InferenceEngine):
    def __init__(self, shard_downloader):
        self.shard_downloader = shard_downloader
        self.shard = None
        self.ensure_shard_calls = []
        
    async def ensure_shard(self, shard):
        self.ensure_shard_calls.append(shard)
        self.shard = shard
        return await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)
        
    async def infer_prompt(self, request_id, shard, prompt):
        return None, None
        
    async def infer_tensor(self, request_id, shard, input_data):
        return None, None
        
    async def sample(self, tensor, temp=0.0):
        return None
        
    # Implement the required abstract methods
    async def decode(self, tokens, **kwargs):
        return "mock_decode_result"
        
    async def encode(self, text):
        return [1, 2, 3]  # Mock token IDs
        
    async def load_checkpoint(self, shard):
        return True


class MockDiscovery(Discovery):
    def __init__(self, peers=None):
        self.peers = peers or []
        
    async def start(self):
        pass
        
    async def stop(self):
        pass
        
    async def discover_peers(self, wait_for_peers=0):
        return self.peers


class MockServer(Server):
    def __init__(self):
        pass
        
    async def start(self):
        pass
        
    async def stop(self):
        pass
        
    @property
    def message_handler(self):
        return None


class MockPartitioningStrategy(PartitioningStrategy):
    def partition(self, topology):
        return []


@pytest.mark.asyncio
async def test_node_setup_peer_download_coordinator(capsys):
    """Test that a node properly sets up a peer download coordinator"""
    
    # Create a mock peer downloader
    peer_downloader = PeerShardDownloader(MockFallbackDownloader())
    
    # Create mock peers
    peer1 = MockPeerHandle("peer1")
    peer2 = MockPeerHandle("peer2")
    
    # Create a mock discovery that will return our peers
    discovery = MockDiscovery(peers=[peer1, peer2])
    
    # Create a test node with our peer downloader that won't create background tasks
    node = NonBackgroundTaskNode(
        _id="test-node",
        server=MockServer(),
        inference_engine=MockInferenceEngine(peer_downloader),
        discovery=discovery,
        shard_downloader=peer_downloader,
        partitioning_strategy=MockPartitioningStrategy()
    )
    
    # Let's spy on the setup_peer_download_coordinator method
    original_method = node.setup_peer_download_coordinator
    setup_spy = AsyncMock(wraps=original_method)
    node.setup_peer_download_coordinator = setup_spy
    
    # Mock collect_topology to avoid None.nodes issue
    # The global fixture handles the AsyncCallback and periodic task creation
    with patch.object(Node, 'collect_topology', return_value=None):
        # Start the node, which should call setup_peer_download_coordinator
        await node.start()
        
        # Verify setup_peer_download_coordinator was called at least once
        assert setup_spy.call_count >= 1
    
    # Verify that peers were set in the downloader
    assert len(peer_downloader.peers) == 2
    
    # Verify coordinator was set
    assert peer_downloader._coordinator_id is not None
    
    # The coordinator should be the node with the lowest ID
    expected_coordinator = min([node.id, peer1.id(), peer2.id()])
    assert peer_downloader._coordinator_id == expected_coordinator
    
    # Check for [PEER DOWNLOAD] logs
    captured = capsys.readouterr()
    assert "[PEER DOWNLOAD]" in captured.out
    assert "Setting up peer download coordination" in captured.out


@pytest.mark.asyncio
async def test_node_updates_peer_downloader_on_peer_changes(capsys):
    """Test that the peer downloader is updated when peers change"""
    
    # Create a mock peer downloader
    peer_downloader = PeerShardDownloader(MockFallbackDownloader())
    
    # Start with no peers
    discovery = MockDiscovery(peers=[])
    
    # Create a test node with our peer downloader that won't create background tasks
    node = NonBackgroundTaskNode(
        _id="test-node",
        server=MockServer(),
        inference_engine=MockInferenceEngine(peer_downloader),
        discovery=discovery,
        shard_downloader=peer_downloader,
        partitioning_strategy=MockPartitioningStrategy()
    )
    
    # Mock collect_topology to avoid None.nodes issue
    # The global fixture handles the AsyncCallback and periodic task creation
    with patch.object(Node, 'collect_topology', return_value=None):
        # Start the node without spawning the periodic task
        await node.start()
    
    # Verify no peers in the downloader
    assert len(peer_downloader.peers) == 0
    
    # Now add some peers
    peer1 = MockPeerHandle("peer1")
    peer2 = MockPeerHandle("peer2")
    discovery.peers = [peer1, peer2]
    
    # Update peers - also patch AsyncCallbackSystem.notify to prevent "coroutine was never awaited" warning
    with patch.object(Node, 'collect_topology', return_value=None):
        with patch('exo.helpers.AsyncCallback.notify', new_callable=AsyncMock) as notify_mock:
            await node.update_peers()
    
    # Verify peers were added to the downloader
    assert len(peer_downloader.peers) == 2
    
    # Check for [PEER DOWNLOAD] logs about peer updates
    captured = capsys.readouterr()
    assert "[PEER DOWNLOAD]" in captured.out
    assert "Peer list updated" in captured.out
    assert "New peers" in captured.out
    
    # Now remove a peer
    discovery.peers = [peer1]
    
    # Update peers again with the same patching to prevent warnings
    with patch.object(Node, 'collect_topology', return_value=None):
        with patch('exo.helpers.AsyncCallback.notify', new_callable=AsyncMock) as notify_mock:
            await node.update_peers()
    
    # Verify peer was removed from the downloader
    assert len(peer_downloader.peers) == 1
    assert peer_downloader.peers[0].id() == "peer1"
    
    # Check for [PEER DOWNLOAD] logs about peer removal
    captured = capsys.readouterr()
    assert "[PEER DOWNLOAD]" in captured.out
    assert "Removed peers" in captured.out


@pytest.mark.asyncio
@patch('exo.download.peer_download.get_repo')
async def test_inference_engine_uses_peer_downloader(mock_get_repo, capsys):
    """Test that the inference engine uses the peer downloader to get shards"""
    
    mock_get_repo.return_value = "test-repo/model"
    
    # Create a mock peer downloader with a spy on ensure_shard
    fallback = MockFallbackDownloader()
    peer_downloader = PeerShardDownloader(fallback)
    ensure_shard_spy = AsyncMock(wraps=peer_downloader.ensure_shard)
    peer_downloader.ensure_shard = ensure_shard_spy
    
    # Create a peer with the model
    peer = MockPeerHandle("peer1", has_model=True, is_complete=True, files=[
        {"path": "model.safetensors", "size": 1024}
    ])
    
    # Create a mock discovery that will return our peer
    discovery = MockDiscovery(peers=[peer])
    
    # Create an inference engine that uses the peer downloader
    inference_engine = MockInferenceEngine(peer_downloader)
    
    # Create a test node that won't create background tasks
    node = NonBackgroundTaskNode(
        _id="test-node",
        server=MockServer(),
        inference_engine=inference_engine,
        discovery=discovery,
        shard_downloader=peer_downloader,
        partitioning_strategy=MockPartitioningStrategy()
    )
    
    # Mock collect_topology to avoid None.nodes issue
    # The global fixture handles the AsyncCallback and periodic task creation
    with patch.object(Node, 'collect_topology', return_value=None):
        # Start the node, which sets up the coordinator
        await node.start()
        
        # Manually set the coordinator to ensure our test works correctly
        peer_downloader.set_coordinator_id("coordinator-id")
    
    # Create a test shard
    shard = Shard("test-model", 0, 1, 2)
    
    # Mock the find_peer_with_model method for testing coordinator behavior
    find_peer_mock = AsyncMock(return_value=peer)
    peer_downloader.find_peer_with_model = find_peer_mock
    
    # Mock the download_model_from_peer method to avoid actual downloading
    peer_downloader.download_model_from_peer = AsyncMock(return_value=Path("/mock/downloads/test-repo--model"))
    
    # Make sure aios.path.exists returns False to force download
    with patch('exo.download.peer_download.aios.path.exists', return_value=False):
        with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
            with patch('exo.download.peer_download.aios.makedirs', new_callable=AsyncMock):
                mock_ensure_dir.return_value = Path("/mock/downloads")
                
                # Ask the inference engine to ensure the shard
                await inference_engine.ensure_shard(shard)
                
                # Verify the peer downloader's ensure_shard was called
                ensure_shard_spy.assert_called_once()
                ensure_shard_spy.assert_called_with(shard, "MockInferenceEngine")
                
                # Verify find_peer_with_model was called (since we're not the coordinator)
                find_peer_mock.assert_called_once()
                
                # Verify logs have [PEER DOWNLOAD] tag
                captured = capsys.readouterr()
                assert "[PEER DOWNLOAD]" in captured.out

@pytest.mark.asyncio
async def test_coordinator_waiting_behavior():
    """Test the wait behavior added for non-coordinator nodes"""
    # We'll test the actual sleep call directly
    with patch('asyncio.sleep', new_callable=AsyncMock) as sleep_mock:
        # Set up test conditions: node is not coordinator and has peers
        is_coordinator = False
        has_peers = True
        
        # Call the sleep method directly - passing the patch correctly
        from exo.orchestration.node import Node
        await Node._coordinator_wait_for_peer_discovery(is_coordinator, has_peers, sleep_mock)
        
        # Verify sleep was called (waiting for coordinator)
        sleep_mock.assert_called_once()


@pytest.mark.asyncio
async def test_end_to_end_coordinator_based_model_download():
    """Test that the non-coordinator nodes properly wait for the coordinator to download a model"""
    # Create a mock topology with two nodes
    coordinator_id = "aaa-coordinator"  # This will be first alphabetically
    non_coordinator_id = "bbb-non-coordinator"
    
    # Create mock peer handles
    # Start with coordinator not having the model
    coordinator_peer = MockPeerHandle(coordinator_id, has_model=False, is_complete=False)
    
    # Create a more sophisticated mock that transitions from not having model 
    # to having an incomplete model to having a complete model
    coordinator_has_model_calls = 0
    
    async def has_model_with_delay(*args, **kwargs):
        nonlocal coordinator_has_model_calls
        coordinator_has_model_calls += 1
        
        response = AsyncMock()
        
        # First call: No model
        if coordinator_has_model_calls == 1:
            response.has_model = False
            response.is_complete = False
        # Second call: Has incomplete model
        elif coordinator_has_model_calls == 2:
            response.has_model = True
            response.is_complete = False
        # Third+ calls: Has complete model
        else:
            response.has_model = True
            response.is_complete = True
            
        return response
    
    # Replace the has_model method on coordinator_peer to simulate coordinator's download state
    coordinator_peer.has_model = has_model_with_delay
    
    # Create test data
    shard = Shard("test-model", 0, 1, 2)
    repo_id = "test-repo/model"
    mock_file = {"path": "model.safetensors", "size": 1024 * 1024}  # 1MB file
    
    # Set up coordinator's files after it "downloads" the model
    async def get_model_file_list_with_delay(*args, **kwargs):
        if coordinator_has_model_calls >= 2:
            return ModelFileListResponse(files=[
                ModelFileInfo(path=mock_file["path"], size=mock_file["size"], hash="")
            ])
        return ModelFileListResponse(files=[])
    
    coordinator_peer.get_model_file_list = get_model_file_list_with_delay
    
    # Create a MockPeerHandle that can identify itself as the non-coordinator
    class EnhancedMockPeerHandle(MockPeerHandle):
        def __init__(self, *args, **kwargs):
            self.own_id = kwargs.pop('own_id', None)
            super().__init__(*args, **kwargs)
    
    # Create mock objects with dependency injection
    fallback_downloader = MockFallbackDownloader()
    
    # Create an enhanced downloader to track specific events
    class TestPeerShardDownloader(PeerShardDownloader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.wait_for_coordinator_called = False
            self.download_from_peer_called = False
            self.fallback_called = False
            
        async def _wait_for_model_on_coordinator(self, *args, **kwargs):
            self.wait_for_coordinator_called = True
            # Call original implementation with shorter timeouts
            return await super()._wait_for_model_on_coordinator(*args, max_wait_seconds=1, poll_interval_seconds=0.1)
            
        async def download_model_from_peer(self, *args, **kwargs):
            self.download_from_peer_called = True
            return Path(f"/mock/downloads/{shard.model_id}")
            
    # Create our test downloader
    downloader = TestPeerShardDownloader(fallback_downloader)
    
    # Set up the peers - non-coordinator with its own ID clearly indicated
    non_coordinator_peer = EnhancedMockPeerHandle(non_coordinator_id, own_id=non_coordinator_id)
    downloader.set_peers([coordinator_peer, non_coordinator_peer])
    downloader.set_coordinator_id(coordinator_id)
    
    # Setup mocks to avoid file system operations
    with patch('exo.download.peer_download.get_repo', return_value=repo_id):
        with patch('exo.download.peer_download.aios.path.exists', return_value=False):
            with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
                with patch('exo.download.peer_download.aios.makedirs', new_callable=AsyncMock):
                    # Setup sleep mock to speed up the test
                    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                        mock_ensure_dir.return_value = Path("/mock/downloads")
                        
                        # Call the ensure_shard method from the perspective of a non-coordinator node
                        result = await downloader.ensure_shard(shard, "MockInferenceEngine")
                        
                        # Verify expected events happened
                        assert downloader.wait_for_coordinator_called, "Should have waited for coordinator"
                        assert coordinator_has_model_calls >= 2, "Should have checked coordinator multiple times" 
                        assert downloader.download_from_peer_called, "Should have downloaded from coordinator"
                        assert not downloader.fallback_called, "Should NOT have used fallback downloader"
                        
                        # Verify the path is correct
                        assert result == Path(f"/mock/downloads/{shard.model_id}")


@pytest.mark.asyncio
async def test_node_wait_for_model_download():
    """Test the waiting mechanism for non-coordinator nodes to find model on coordinator"""
    # Create mock peers and downloader
    peer = MockPeerHandle("coordinator", has_model=False)
    downloader = MockPeerShardDownloader(MockFallbackDownloader())
    
    # Create a mock inference engine
    inference_engine = MockInferenceEngine(downloader)
    
    # Create a test node with the mock objects
    node = NonBackgroundTaskNode(
        _id="non-coordinator",
        server=MockServer(),
        shard_downloader=downloader,
        inference_engine=inference_engine,
        discovery=MockDiscovery([peer]),
        partitioning_strategy=MockPartitioningStrategy()
    )
    
    # Create test shard
    shard = Shard("test-model", 0, 1, 2)
    
    # Setting up mock has_model to return False initially, then True after some calls
    call_count = 0
    
    async def mock_has_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        # Return True after 3 calls to simulate coordinator eventually having the model
        has_model = call_count >= 3
        
        response = AsyncMock()
        response.has_model = has_model
        response.is_complete = has_model
        return response
    
    # Replace the has_model method
    peer.has_model = mock_has_model
    
    # Create a mock sleep function to avoid actual sleep
    sleep_mock = AsyncMock()
    
    # Test the wait_for_model_download method
    result = await node._wait_for_model_download(
        peer, "test-repo", "main", max_wait_seconds=5, sleep_func=sleep_mock
    )
    
    # Verify results
    assert result is True  # Successfully found model
    assert call_count >= 3  # Made at least 3 calls to has_model
    assert sleep_mock.call_count >= 2  # Called sleep between poll attempts
    

@pytest.mark.asyncio
async def test_non_coordinator_never_uses_fallback():
    """Test that non-coordinator nodes NEVER use the fallback downloader"""
    # Create a mock topology with two nodes
    coordinator_id = "aaa-coordinator"
    non_coordinator_id = "bbb-non-coordinator"
    
    # Create mock peer handles
    # Coordinator will never have the model (to simulate permanent failure)
    coordinator_peer = MockPeerHandle(coordinator_id, has_model=False, is_complete=False)
    
    # Create test data
    shard = Shard("test-model", 0, 1, 2)
    repo_id = "test-repo/model"
    
    # Create a MockPeerHandle that can identify itself as the non-coordinator
    class EnhancedMockPeerHandle(MockPeerHandle):
        def __init__(self, *args, **kwargs):
            self.own_id = kwargs.pop('own_id', None)
            super().__init__(*args, **kwargs)
    
    # Create mock objects with dependency injection
    fallback_downloader = MockFallbackDownloader()
    
    # Create a spy on the fallback_downloader.ensure_shard method
    original_fallback_ensure = fallback_downloader.ensure_shard
    fallback_ensure_spy = AsyncMock(wraps=original_fallback_ensure)
    fallback_downloader.ensure_shard = fallback_ensure_spy
    
    # Create our test downloader
    downloader = PeerShardDownloader(fallback_downloader)
    
    # Set up the peers with proper identity
    non_coordinator_peer = EnhancedMockPeerHandle(non_coordinator_id, own_id=non_coordinator_id)
    downloader.set_peers([coordinator_peer, non_coordinator_peer])
    downloader.set_coordinator_id(coordinator_id)
    
    # Make sure our node knows it's NOT the coordinator
    def mock_am_i_coordinator():
        return False
        
    # Ensure the _wait_for_model_on_coordinator method returns quickly
    with patch.object(downloader, '_wait_for_model_on_coordinator', new_callable=AsyncMock) as mock_wait:
        # Configure the wait method to return False (timeout waiting for coordinator)
        mock_wait.return_value = False
        
        # Setup mocks to avoid file system operations
        with patch('exo.download.peer_download.get_repo', return_value=repo_id):
            with patch('exo.download.peer_download.aios.path.exists', return_value=False):
                with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
                    with patch('exo.download.peer_download.aios.makedirs', new_callable=AsyncMock):
                        # Setup sleep mock to speed up the test
                        with patch('asyncio.sleep', new_callable=AsyncMock):
                            mock_ensure_dir.return_value = Path("/mock/downloads")
                            
                            # Try to call ensure_shard from non-coordinator perspective
                            result = await downloader.ensure_shard(shard, "MockInferenceEngine")
                            
                            # Verify the fallback_downloader was NEVER called
                            # This is the key assertion - non-coordinators should NEVER use the fallback
                            fallback_ensure_spy.assert_not_called()
                            
                            # The result should still be a valid path (but the model will be missing/incomplete)
                            assert isinstance(result, Path)
                            assert "mock/downloads" in str(result)


if __name__ == "__main__":
    pytest.main(["-xvs", "test_peer_download_coordination.py"])