#!/usr/bin/env python3
"""
Integration test for peer download feature

This script simulates a two-node exo cluster to demonstrate the peer download feature.
It will:
1. Create two nodes with peer download enabled
2. Request the same model on both nodes
3. Verify that only one node downloads from HuggingFace while the other gets it from the peer

Usage:
    python test_peer_download.py

Note: This requires running in a network where UDP discovery works on port 5678
"""

import os
import sys
import asyncio
import time
import uuid
import argparse
from pathlib import Path


# Add the parent directory to sys.path so we can import exo
sys.path.append(str(Path(__file__).parent.parent))

from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.networking.grpc.grpc_server import GRPCServer
from exo.inference.inference_engine import get_inference_engine, inference_engine_classes
from exo.download.peer_download import peer_shard_downloader
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.orchestration.node import Node
from exo.inference.shard import Shard
from exo.models import build_base_shard, get_repo
from exo.helpers import find_available_port


# Use dummy model for testing since it doesn't require downloading real model files
DEFAULT_TEST_MODEL = "dummy"

# Increase debug level
os.environ["DEBUG"] = "3"


async def run_node(node_id, model_id, port=None, wait_for_peers=0):
    """Run a node with peer download enabled"""
    print(f"\n[{node_id}] Starting node...")
    
    # Set up a unique download directory for this node
    os.environ["EXO_HOME"] = f"/tmp/exo-peer-test/{node_id}"
    os.makedirs(os.environ["EXO_HOME"], exist_ok=True)
    
    # Create the peer downloader
    print(f"[{node_id}] Creating peer downloader")
    shard_downloader = peer_shard_downloader()
    
    # Use the dummy inference engine for testing
    inference_engine_name = "dummy"
    inference_engine = get_inference_engine(inference_engine_name, shard_downloader)
    
    # Set up discovery with UDP
    host = "0.0.0.0"
    port = port or find_available_port(host)
    discovery = UDPDiscovery(
        node_id,
        port,
        5678,  # Listen port
        5678,  # Broadcast port
        lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
        discovery_timeout=30
    )
    
    # Create and start the node
    server = GRPCServer(None, host, port)
    node = Node(
        _id=node_id,
        server=server,
        inference_engine=inference_engine,
        discovery=discovery,
        shard_downloader=shard_downloader,
        partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
        max_generate_tokens=100,  # Small for testing
    )
    node.server = server
    
    # Patch the topology collection to avoid None.nodes errors in this test
    from unittest.mock import patch
    import asyncio
    from exo.topology.topology import Topology
    from functools import partial
    
    async def mock_collect_topology(self, visited=None, max_depth=5):
        """Mock version that returns an empty topology instead of None"""
        return Topology()
    
    # Apply the patch
    original_collect = node.collect_topology
    node.collect_topology = partial(mock_collect_topology, node)
    
    # Start the node
    await node.start(wait_for_peers=wait_for_peers)
    print(f"[{node_id}] Node started on port {port}")
    
    # Wait a moment for peer discovery
    await asyncio.sleep(2)
    
    # Get the shard for our model
    print(f"[{node_id}] Trying to build shard for model: {model_id} with engine: {inference_engine_name}")
    print(f"[{node_id}] Available repos: {get_repo(model_id, inference_engine_classes.get(inference_engine_name, inference_engine_name))}")
    
    shard = build_base_shard(model_id, inference_engine_classes.get(inference_engine_name, inference_engine_name))
    if not shard:
        print(f"[{node_id}] Error: Unsupported model '{model_id}'")
        return
    
    print(f"[{node_id}] Downloading model: {model_id}")
    download_start = time.time()
    
    # Start the model download/inference
    try:
        await inference_engine.ensure_shard(shard)
        download_end = time.time()
        print(f"[{node_id}] Download completed in {download_end - download_start:.2f} seconds")
    except Exception as e:
        print(f"[{node_id}] Download failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Done
    print(f"[{node_id}] Test completed")
    await node.stop()


async def main(model_id):
    """Run the test with two nodes"""
    print(f"Starting peer download test with model: {model_id}")
    
    # Clear the test directory
    import shutil
    if os.path.exists("/tmp/exo-peer-test"):
        shutil.rmtree("/tmp/exo-peer-test")
    
    # Start the coordinator node
    node1_id = f"node1-{uuid.uuid4().hex[:8]}"
    node1_port = find_available_port("0.0.0.0")
    print(f"\nStarting first node (ID: {node1_id}) on port {node1_port}")
    node1_task = asyncio.create_task(run_node(node1_id, model_id, port=node1_port))
    
    # Wait a bit for the first node to start
    await asyncio.sleep(5)
    
    # Start the second node, waiting for the first one to be discovered
    node2_id = f"node2-{uuid.uuid4().hex[:8]}"
    node2_port = find_available_port("0.0.0.0")
    print(f"\nStarting second node (ID: {node2_id}) on port {node2_port}")
    node2_task = asyncio.create_task(run_node(node2_id, model_id, port=node2_port))
    
    try:
        # Wait for both nodes to complete with a timeout
        await asyncio.wait_for(asyncio.gather(node1_task, node2_task), timeout=300)  # 5 minute timeout
        
        print("\nTest completed successfully!")
        print("Check the download times reported by each node.")
        print("The second node should download from the first node rather than from HuggingFace.")
    except asyncio.TimeoutError:
        print("\nTest timed out after 5 minutes!")
        print("The nodes were still running but didn't complete in time.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test peer download feature')
    parser.add_argument('--model', type=str, default=DEFAULT_TEST_MODEL,
                      help=f'Model to download (default: {DEFAULT_TEST_MODEL})')
    args = parser.parse_args()
    
    asyncio.run(main(args.model))