import grpc
from concurrent import futures
import numpy as np
from asyncio import CancelledError

import platform

from . import node_service_pb2
from . import node_service_pb2_grpc
from exo import DEBUG
from exo.inference.shard import Shard
from exo.orchestration import Node
import json

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
  import mlx.core as mx
else:
  import numpy as mx


class GRPCServer(node_service_pb2_grpc.NodeServiceServicer):
  def __init__(self, node: Node, host: str, port: int):
    self.node = node
    self.host = host
    self.port = port
    self.server = None

  async def start(self) -> None:
    self.server = grpc.aio.server(
      futures.ThreadPoolExecutor(max_workers=32),
      options=[
        ("grpc.max_metadata_size", 32*1024*1024),
        ("grpc.max_send_message_length", 256*1024*1024),
        ("grpc.max_receive_message_length", 256*1024*1024),
        ("grpc.keepalive_time_ms", 10000),
        ("grpc.keepalive_timeout_ms", 5000),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.http2.min_time_between_pings_ms", 10000),
        ("grpc.http2.min_ping_interval_without_data_ms", 5000),
        ("grpc.max_concurrent_streams", 100),
        ("grpc.tcp_nodelay", 1),
        ("grpc.optimization_target", "throughput"),
      ],
    )
    node_service_pb2_grpc.add_NodeServiceServicer_to_server(self, self.server)
    listen_addr = f"{self.host}:{self.port}"
    self.server.add_insecure_port(listen_addr)
    await self.server.start()
    if DEBUG >= 1: print(f"Server started, listening on {listen_addr}")

  async def stop(self) -> None:
    if self.server:
      try:
        await self.server.stop(grace=5)
        await self.server.wait_for_termination()
      except CancelledError:
        pass
      if DEBUG >= 1: print("Server stopped and all connections are closed")

  async def SendPrompt(self, request, context):
    shard = Shard(
      model_id=request.shard.model_id,
      start_layer=request.shard.start_layer,
      end_layer=request.shard.end_layer,
      n_layers=request.shard.n_layers,
    )
    prompt = request.prompt
    request_id = request.request_id
    inference_state = None if request.inference_state is None else self.deserialize_inference_state(request.inference_state)
    result = await self.node.process_prompt(shard, prompt, request_id, inference_state)
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

    inference_state = None if request.inference_state is None else self.deserialize_inference_state(request.inference_state)

    result = await self.node.process_tensor(shard, tensor, request_id, inference_state)
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
      node_id: node_service_pb2.PeerConnections(connections=[node_service_pb2.PeerConnection(to_id=conn.to_id, description=conn.description) for conn in connections])
      for node_id, connections in topology.peer_graph.items()
    }
    if DEBUG >= 5: print(f"CollectTopology {max_depth=} {visited=} {nodes=} {peer_graph=}")
    return node_service_pb2.Topology(nodes=nodes, peer_graph=peer_graph)

  async def SendResult(self, request, context):
    request_id = request.request_id
    result = request.result
    is_finished = request.is_finished
    img = request.tensor
    if DEBUG >= 5: print(f"Received SendResult request: {request_id=} {result=} {is_finished=}")
    result = list(result)
    if len(img.tensor_data) > 0:
      result = np.frombuffer(img.tensor_data, dtype=np.dtype(img.dtype)).reshape(img.shape)
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
    
  async def HasModel(self, request, context):
    """Check if this node has a given model downloaded"""
    import os
    import aiofiles.os as aios
    from pathlib import Path
    from exo.download.new_shard_download import ensure_downloads_dir
    
    try:
      repo_id = request.repo_id
      revision = request.revision
      
      # Convert repo_id to local filesystem path
      model_dir = await ensure_downloads_dir() / repo_id.replace("/", "--")
      
      has_model = await aios.path.exists(model_dir)
      available_files = []
      is_complete = False
      
      if has_model:
        # Get all files in the model directory
        available_files = []
        for root, dirs, files in os.walk(model_dir):
          rel_root = str(Path(root).relative_to(model_dir))
          for file in files:
            if not file.endswith(".partial"):  # Skip partial downloads
              if rel_root == ".":
                available_files.append(file)
              else:
                available_files.append(os.path.join(rel_root, file))
                
        # Check for specific config files to determine if model is complete
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        has_required_files = any(file in available_files for file in required_files)
        
        # Look for model weight files (could be in multiple formats)
        has_weight_file = any(
            file.endswith(".safetensors") or 
            file.endswith(".bin") or
            file.endswith(".pt") or
            file.endswith(".gguf")
            for file in available_files
        )
        
        # Log details about what we found for debugging
        if DEBUG >= 2:
            print(f"[GRPC SERVER] For model {repo_id}:")
            print(f"[GRPC SERVER]   Has required files: {has_required_files}")
            print(f"[GRPC SERVER]   Has weight file: {has_weight_file}")
            print(f"[GRPC SERVER]   Available files: {available_files}")
        
        # Consider the model complete if it has required config files and at least one weight file
        is_complete = has_required_files and has_weight_file
      
      return node_service_pb2.HasModelResponse(
        has_model=has_model,
        is_complete=is_complete,
        available_files=available_files
      )
    except Exception as e:
      import traceback
      traceback.print_exc()
      context.abort(grpc.StatusCode.INTERNAL, str(e))
      
  async def GetModelFileList(self, request, context):
    """Return a list of files for a given model"""
    import os
    import aiofiles.os as aios
    from pathlib import Path
    from exo.download.new_shard_download import ensure_downloads_dir
    
    try:
      repo_id = request.repo_id
      revision = request.revision
      allow_patterns = list(request.allow_patterns)
      
      # Convert repo_id to local filesystem path
      model_dir = await ensure_downloads_dir() / repo_id.replace("/", "--")
      
      if not await aios.path.exists(model_dir):
        return node_service_pb2.ModelFileListResponse(files=[])
        
      # Get all files in the model directory
      files = []
      for root, dirs, filenames in os.walk(model_dir):
        rel_root = Path(root).relative_to(model_dir)
        for filename in filenames:
          if filename.endswith(".partial"):
            continue  # Skip partial downloads
            
          file_path = os.path.join(str(rel_root), filename) if str(rel_root) != "." else filename
          
          # Apply pattern filtering if patterns are provided
          if allow_patterns and not any(file_path.startswith(pattern.strip("*")) for pattern in allow_patterns):
            if not any(pattern == "*" for pattern in allow_patterns):
              continue
          
          # Get file size
          file_stat = await aios.stat(Path(root) / filename)
          file_size = file_stat.st_size
          
          # Here we could calculate file hash but it's expensive
          # For now we'll return an empty hash
          file_hash = ""
          
          files.append(node_service_pb2.ModelFileInfo(
            path=file_path,
            size=file_size,
            hash=file_hash
          ))
      
      return node_service_pb2.ModelFileListResponse(files=files)
    except Exception as e:
      import traceback
      traceback.print_exc()
      context.abort(grpc.StatusCode.INTERNAL, str(e))
      
  async def GetModelFile(self, request, context):
    """Stream a model file in chunks"""
    import os
    import aiofiles
    import aiofiles.os as aios
    from pathlib import Path
    from exo.download.new_shard_download import ensure_downloads_dir
    
    try:
      repo_id = request.repo_id
      revision = request.revision
      file_path = request.file_path
      offset = request.offset
      
      # Convert repo_id to local filesystem path
      model_dir = await ensure_downloads_dir() / repo_id.replace("/", "--")
      full_path = model_dir / file_path
      
      if not await aios.path.exists(full_path):
        context.abort(grpc.StatusCode.NOT_FOUND, f"File {file_path} not found")
        return
        
      # Get file size
      file_stat = await aios.stat(full_path)
      file_size = file_stat.st_size
      
      if offset >= file_size:
        context.abort(grpc.StatusCode.OUT_OF_RANGE, f"Offset {offset} is beyond file size {file_size}")
        return
        
      # Read and stream file in chunks
      chunk_size = 1024 * 1024  # 1MB chunks
      
      async with aiofiles.open(full_path, 'rb') as f:
        await f.seek(offset)
        
        while True:
          data = await f.read(chunk_size)
          if not data:
            break
            
          curr_offset = offset
          offset += len(data)
          
          yield node_service_pb2.FileChunk(
            data=data,
            offset=curr_offset,
            total_size=file_size
          )
          
    except Exception as e:
      import traceback
      traceback.print_exc()
      context.abort(grpc.StatusCode.INTERNAL, str(e))

  def deserialize_inference_state(self, inference_state_proto: node_service_pb2.InferenceState) -> dict:
    inference_state = {}

    for k, tensor_data in inference_state_proto.tensor_data.items():
      np_array = np.frombuffer(tensor_data.tensor_data, dtype=tensor_data.dtype).reshape(tensor_data.shape)
      inference_state[k] = mx.array(np_array)

    for k, tensor_list in inference_state_proto.tensor_list_data.items():
      inference_state[k] = [mx.array(np.frombuffer(tensor.tensor_data, dtype=tensor.dtype).reshape(tensor.shape)) for tensor in tensor_list.tensors]

    if inference_state_proto.other_data_json:
      other_data = json.loads(inference_state_proto.other_data_json)
      inference_state.update(other_data)

    return inference_state
