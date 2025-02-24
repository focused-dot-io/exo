import pytest
import asyncio
import uuid
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from exo.inference.shard import Shard
from exo.networking.peer_handle import PeerHandle
from exo.download.peer_download import PeerShardDownloader
from exo.download.shard_download import ShardDownloader
from exo.orchestration.node import Node
from exo.networking.server import Server
from exo.networking.discovery import Discovery
from exo.inference.inference_engine import InferenceEngine
from exo.topology.partitioning_strategy import PartitioningStrategy

# Import MockPeerHandle from the unit tests
from exo.download.test_peer_download import MockPeerHandle, MockFallbackDownloader


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
async def test_node_setup_peer_download_coordinator():
    """Test that a node properly sets up a peer download coordinator"""
    
    # Create a mock peer downloader
    peer_downloader = PeerShardDownloader(MockFallbackDownloader())
    
    # Create mock peers
    peer1 = MockPeerHandle("peer1")
    peer2 = MockPeerHandle("peer2")
    
    # Create a mock discovery that will return our peers
    discovery = MockDiscovery(peers=[peer1, peer2])
    
    # Create a node with our peer downloader
    node = Node(
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


@pytest.mark.asyncio
async def test_node_updates_peer_downloader_on_peer_changes():
    """Test that the peer downloader is updated when peers change"""
    
    # Create a mock peer downloader
    peer_downloader = PeerShardDownloader(MockFallbackDownloader())
    
    # Start with no peers
    discovery = MockDiscovery(peers=[])
    
    # Create a node with our peer downloader
    node = Node(
        _id="test-node",
        server=MockServer(),
        inference_engine=MockInferenceEngine(peer_downloader),
        discovery=discovery,
        shard_downloader=peer_downloader,
        partitioning_strategy=MockPartitioningStrategy()
    )
    
    # Start the node
    await node.start()
    
    # Verify no peers in the downloader
    assert len(peer_downloader.peers) == 0
    
    # Now add some peers
    peer1 = MockPeerHandle("peer1")
    peer2 = MockPeerHandle("peer2")
    discovery.peers = [peer1, peer2]
    
    # Update peers
    await node.update_peers()
    
    # Verify peers were added to the downloader
    assert len(peer_downloader.peers) == 2
    
    # Now remove a peer
    discovery.peers = [peer1]
    
    # Update peers again
    await node.update_peers()
    
    # Verify peer was removed from the downloader
    assert len(peer_downloader.peers) == 1
    assert peer_downloader.peers[0].id() == "peer1"


@pytest.mark.asyncio
@patch('exo.download.peer_download.get_repo')
async def test_inference_engine_uses_peer_downloader(mock_get_repo):
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
    
    # Create a node
    node = Node(
        _id="test-node",
        server=MockServer(),
        inference_engine=inference_engine,
        discovery=discovery,
        shard_downloader=peer_downloader,
        partitioning_strategy=MockPartitioningStrategy()
    )
    
    # Mock collect_topology to avoid None.nodes issue
    with patch.object(Node, 'collect_topology', return_value=None):
        # Start the node, which sets up the coordinator
        await node.start()
        
        # Manually set the coordinator to ensure our test works correctly
        peer_downloader.set_coordinator_id("coordinator-id")
    
    # Create a test shard
    shard = Shard("test-model", 0, 1, 2)
    
    # Mock the download_model_from_peer method to avoid actual downloading
    peer_downloader.download_model_from_peer = AsyncMock(return_value=Path("/mock/downloads/test-repo--model"))
    
    # Make sure aios.path.exists returns False to force download
    with patch('exo.download.peer_download.aios.path.exists', return_value=False):
        with patch('exo.download.peer_download.ensure_downloads_dir', new_callable=AsyncMock) as mock_ensure_dir:
            mock_ensure_dir.return_value = Path("/mock/downloads")
            
            # Ask the inference engine to ensure the shard
            await inference_engine.ensure_shard(shard)
            
            # Verify the peer downloader's ensure_shard was called
            ensure_shard_spy.assert_called_once()
            ensure_shard_spy.assert_called_with(shard, "MockInferenceEngine")
            
            # If the node ID is not the coordinator, it should try to find a peer with the model
            if node.id != peer_downloader._coordinator_id:
                # Check if peer.has_model was called
                assert len(peer.has_model_calls) > 0


if __name__ == "__main__":
    pytest.main(["-xvs", "test_peer_download_coordination.py"])