import numpy as np
from exo.inference.shard import Shard
from exo.orchestration import Node
from exo import DEBUG
from . import node_service_pb2
from . import node_service_pb2_grpc
import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator
from exo.download.hf.hf_helpers import get_local_snapshot_dir, get_weight_map, get_allow_patterns
from exo.models import get_repo, get_shard_path
import traceback
from typing import Dict, List, Optional, Tuple
import grpc
import grpc.aio

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

class NodeServiceHandler(node_service_pb2_grpc.NodeServiceServicer):
    def __init__(self, node: Node):
        self.node = node

    async def SendPrompt(self, request, context):
        shard = Shard(
            model_id=request.shard.model_id,
            start_layer=request.shard.start_layer,
            end_layer=request.shard.end_layer,
            n_layers=request.shard.n_layers,
        )
        prompt = request.prompt
        request_id = request.request_id
        result = await self.node.process_prompt(shard, prompt, request_id)
        if DEBUG >= 5: print(f"SendPrompt {shard=} {prompt=} {request_id=} result: {result}")
        tensor_data = result.tobytes() if result is not None else None
        return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype)) if result is not None else node_service_pb2.Tensor()

    async def SendTensor(self, request, context):
        shard = Shard(
            model_id=request.shard.model_id,
            start_layer=request.shard.start_layer,
            end_layer=request.shard.end_layer,
            n_layers=request.shard.n_layers,
        )
        tensor = np.frombuffer(request.tensor.tensor_data, dtype=np.dtype(request.tensor.dtype)).reshape(request.tensor.shape)
        request_id = request.request_id

        result = await self.node.process_tensor(shard, tensor, request_id)
        if DEBUG >= 5: print(f"SendTensor tensor {shard=} {tensor=} {request_id=} result: {result}")
        tensor_data = result.tobytes() if result is not None else None
        return node_service_pb2.Tensor(tensor_data=tensor_data, shape=result.shape, dtype=str(result.dtype)) if result is not None else node_service_pb2.Tensor()
    
    async def SendExample(self, request, context):
        shard = Shard(
            model_id=request.shard.model_id,
            start_layer=request.shard.start_layer,
            end_layer=request.shard.end_layer,
            n_layers=request.shard.n_layers,
        )
        example = np.frombuffer(request.example.tensor_data, dtype=np.dtype(request.example.dtype)).reshape(request.example.shape)
        target = np.frombuffer(request.target.tensor_data, dtype=np.dtype(request.target.dtype)).reshape(request.target.shape)
        length = np.frombuffer(request.length.tensor_data, dtype=np.dtype(request.length.dtype)).reshape(request.length.shape)
        train = request.train
        request_id = request.request_id

        if train and not shard.is_first_layer():
            loss, grad = await self.node.process_example(shard, example, target, length, train, request_id)
            tensor_data = grad.tobytes()
            grad_tensor = node_service_pb2.Tensor(tensor_data=tensor_data, shape=grad.shape, dtype=str(grad.dtype))
            return node_service_pb2.Loss(loss=loss, grads=grad_tensor)
        else:
            loss = await self.node.process_example(shard, example, target, length, train, request_id)
            return node_service_pb2.Loss(loss=loss, grads=None)
        
    async def CollectTopology(self, request, context):
        max_depth = request.max_depth
        visited = set(request.visited)
        topology = self.node.current_topology
        nodes = {
            node_id:
                node_service_pb2.DeviceCapabilities(
                    model=cap.model,
                    chip=cap.chip,
                    memory=cap.memory,
                    flops=node_service_pb2.DeviceFlops(fp32=cap.flops.fp32, fp16=cap.flops.fp16, int8=cap.flops.int8),
                )
            for node_id, cap in topology.nodes.items()
        }
        peer_graph = {
            node_id: node_service_pb2.PeerConnections(
                connections=[
                    node_service_pb2.PeerConnection(to_id=conn.to_id, description=conn.description)
                    for conn in connections
                ]
            )
            for node_id, connections in topology.peer_graph.items()
        }
        if DEBUG >= 5: print(f"CollectTopology {max_depth=} {visited=} {nodes=} {peer_graph=}")
        return node_service_pb2.Topology(nodes=nodes, peer_graph=peer_graph)

    async def SendResult(self, request, context):
        request_id = request.request_id
        result = request.result
        is_finished = request.is_finished
        if DEBUG >= 5: print(f"Received SendResult request: {request_id=} {result=} {is_finished=}")
        self.node.on_token.trigger_all(request_id, result, is_finished)
        return node_service_pb2.Empty()

    async def SendOpaqueStatus(self, request, context):
        request_id = request.request_id
        status = request.status
        if DEBUG >= 8: print(f"Received SendOpaqueStatus request: {request_id=} {status=}")
        self.node.on_opaque_status.trigger_all(request_id, status)
        return node_service_pb2.Empty()

    async def HealthCheck(self, request, context):
        return node_service_pb2.HealthCheckResponse(is_healthy=True) 

    async def GetShardStatus(
        self,
        request: node_service_pb2.GetShardStatusRequest,
        context
    ) -> node_service_pb2.GetShardStatusResponse:
        try:
            if DEBUG >= 2:
                print(f"[Node Service] Checking if we have shard {Shard.from_proto(request.shard)} from repo")
                print(f"[Node Service] Node ID: {self.node.id}")

            # Check if we have the shard locally
            shard = Shard.from_proto(request.shard)
            repo = get_repo(shard.model_id, request.inference_engine_name)
            local_path = await get_shard_path(repo, shard)

            if local_path and local_path.exists():
                if DEBUG >= 2:
                    print(f"[Node Service] Found shard at {local_path}")
                return node_service_pb2.GetShardStatusResponse(
                    has_shard=True,
                    local_path=str(local_path),
                    file_size=local_path.stat().st_size
                )
            else:
                if DEBUG >= 2:
                    print(f"[Node Service] Shard not found at {local_path}")
                return node_service_pb2.GetShardStatusResponse(
                    has_shard=False,
                    local_path="",
                    file_size=0
                )

        except Exception as e:
            if DEBUG >= 2:
                print(f"[Node Service] Error in GetShardStatus: {str(e)}")
                print(f"[Node Service] Error type: {type(e)}")
                traceback.print_exc()
            return node_service_pb2.GetShardStatusResponse(
                has_shard=False,
                local_path="",
                file_size=0
            )

    async def TransferShard(
        self,
        request_iterator: AsyncIterator[node_service_pb2.ShardChunk],
        context
    ) -> AsyncIterator[node_service_pb2.TransferStatus]:
        try:
            # Get initial metadata request
            request = await request_iterator.__anext__()
            if not request.HasField("metadata"):
                yield node_service_pb2.TransferStatus(
                    status=node_service_pb2.TransferStatus.ERROR,
                    error_message="First message must contain metadata"
                )
                return
                
            metadata = request.metadata
            shard = Shard(
                model_id=metadata.shard.model_id,
                start_layer=metadata.shard.start_layer,
                end_layer=metadata.shard.end_layer,
                n_layers=metadata.shard.n_layers,
            )
            repo_name = get_repo(shard.model_id, metadata.inference_engine_name)
            
            if DEBUG >= 2:
                print(f"[Node Service] Starting transfer of shard {shard}")
            
            # Find shard file
            snapshot_dir = await get_local_snapshot_dir(repo_name)
            if not snapshot_dir:
                if DEBUG >= 2:
                    print(f"[Node Service] No snapshot directory found for {repo_name}")
                yield node_service_pb2.TransferStatus(
                    status=node_service_pb2.TransferStatus.ERROR,
                    error_message="Shard not found locally"
                )
                return

            # Get the model file
            model_file = snapshot_dir / "model.safetensors"
            if not model_file.exists():
                if DEBUG >= 2:
                    print(f"[Node Service] Model file not found at {model_file}")
                yield node_service_pb2.TransferStatus(
                    status=node_service_pb2.TransferStatus.ERROR,
                    error_message=f"Model file not found at {model_file}"
                )
                return
                
            file_size = model_file.stat().st_size
            bytes_sent = 0
            
            if DEBUG >= 2:
                print(f"[Node Service] Starting transfer of {model_file} (size: {file_size})")
            
            # Send initial response with file info
            yield node_service_pb2.TransferStatus(
                status=node_service_pb2.TransferStatus.OK,
                bytes_received=0
            )
            
            # Send file in chunks
            with open(model_file, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                        
                    chunk_data = node_service_pb2.ShardChunk(
                        chunk_data=chunk,
                        offset=bytes_sent,
                        is_last=False
                    )
                    await request_iterator.asend(chunk_data)
                    
                    bytes_sent += len(chunk)
                    if DEBUG >= 3:
                        print(f"[Node Service] Sent chunk of size {len(chunk)} at offset {bytes_sent}")
                    
                    yield node_service_pb2.TransferStatus(
                        status=node_service_pb2.TransferStatus.OK,
                        bytes_received=bytes_sent
                    )
                    
                    await asyncio.sleep(0)  # Yield control
            
            # Send final chunk
            await request_iterator.asend(node_service_pb2.ShardChunk(
                chunk_data=b"",
                offset=bytes_sent,
                is_last=True
            ))
                    
            # Send final status
            if DEBUG >= 2:
                print(f"[Node Service] Completed transfer of shard {shard}")
            yield node_service_pb2.TransferStatus(
                status=node_service_pb2.TransferStatus.OK,
                bytes_received=file_size
            )
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Node Service] Error in TransferShard: {e}")
            yield node_service_pb2.TransferStatus(
                status=node_service_pb2.TransferStatus.ERROR,
                error_message=str(e)
            ) 