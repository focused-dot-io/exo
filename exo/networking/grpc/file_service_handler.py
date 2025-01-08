import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator
from exo.networking.grpc.file_service_pb2 import (
    GetShardStatusRequest, GetShardStatusResponse,
)
from exo.networking.grpc.node_service_pb2 import (
    ShardChunk,
    TransferStatus,
    Shard
)
from exo.download.shard import Shard as ShardModel
from exo.networking.grpc.file_service_pb2_grpc import FileServiceServicer
from exo.download.hf.hf_helpers import get_local_snapshot_dir, get_weight_map, get_allow_patterns
from exo.models import get_repo
from exo.helpers import DEBUG

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

class FileServiceHandler(FileServiceServicer):
    async def GetShardStatus(
        self,
        request: GetShardStatusRequest,
        context
    ) -> GetShardStatusResponse:
        try:
            shard = ShardModel.from_proto(request.shard)
            repo_name = get_repo(shard.model_id, request.inference_engine_name)
            
            if DEBUG >= 2:
                print(f"[File Service] Checking if we have shard {shard} from repo {repo_name}")
            
            # Check if we have the shard locally
            snapshot_dir = await get_local_snapshot_dir(repo_name)
            if not snapshot_dir:
                if DEBUG >= 2:
                    print(f"[File Service] No snapshot directory found for {repo_name}")
                return GetShardStatusResponse(has_shard=False)

            # Get weight map to find shard files
            weight_map = await get_weight_map(repo_name)
            if not weight_map:
                if DEBUG >= 2:
                    print(f"[File Service] No weight map found for {repo_name}")
                return GetShardStatusResponse(has_shard=False)

            # Get patterns for this shard
            allow_patterns = get_allow_patterns(weight_map, shard)
            if not allow_patterns:
                if DEBUG >= 2:
                    print(f"[File Service] No patterns found for shard {shard}")
                return GetShardStatusResponse(has_shard=False)

            # Find the main model file
            model_file = snapshot_dir / "model.safetensors"
            if not model_file.exists():
                if DEBUG >= 2:
                    print(f"[File Service] Model file not found at {model_file}")
                return GetShardStatusResponse(has_shard=False)

            if DEBUG >= 2:
                print(f"[File Service] Found shard {shard} at {model_file}")
                
            return GetShardStatusResponse(
                has_shard=True,
                local_path=str(model_file),
                file_size=model_file.stat().st_size
            )
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"[File Service] Error in GetShardStatus: {e}")
            return GetShardStatusResponse(has_shard=False)

    async def TransferShard(
        self,
        request_iterator: AsyncIterator[ShardChunk],
        context
    ) -> AsyncIterator[TransferStatus]:
        try:
            # Get initial metadata request
            request = await request_iterator.__anext__()
            if not request.HasField("metadata"):
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message="First message must contain metadata"
                )
                return
                
            metadata = request.metadata
            shard = ShardModel.from_proto(metadata.shard)
            repo_name = get_repo(shard.model_id, metadata.inference_engine_name)
            
            if DEBUG >= 2:
                print(f"[File Service] Starting transfer of shard {shard}")
            
            # Find shard file
            snapshot_dir = await get_local_snapshot_dir(repo_name)
            if not snapshot_dir:
                if DEBUG >= 2:
                    print(f"[File Service] No snapshot directory found for {repo_name}")
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message="Shard not found locally"
                )
                return

            # Get the model file
            model_file = snapshot_dir / "model.safetensors"
            if not model_file.exists():
                if DEBUG >= 2:
                    print(f"[File Service] Model file not found at {model_file}")
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message=f"Model file not found at {model_file}"
                )
                return
                
            file_size = model_file.stat().st_size
            bytes_sent = 0
            
            if DEBUG >= 2:
                print(f"[File Service] Starting transfer of {model_file} (size: {file_size})")
            
            # Send initial response with file info
            yield TransferStatus(
                status=TransferStatus.OK,
                bytes_received=0
            )
            
            # Send file in chunks
            with open(model_file, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                        
                    chunk_data = ShardChunk(
                        chunk_data=chunk,
                        offset=bytes_sent,
                        is_last=False
                    )
                    await request_iterator.asend(chunk_data)
                    
                    bytes_sent += len(chunk)
                    if DEBUG >= 3:
                        print(f"[File Service] Sent chunk of size {len(chunk)} at offset {bytes_sent}")
                    
                    yield TransferStatus(
                        status=TransferStatus.OK,
                        bytes_received=bytes_sent
                    )
                    
                    await asyncio.sleep(0)  # Yield control
            
            # Send final chunk
            await request_iterator.asend(ShardChunk(
                chunk_data=b"",
                offset=bytes_sent,
                is_last=True
            ))
                    
            # Send final status
            if DEBUG >= 2:
                print(f"[File Service] Completed transfer of shard {shard}")
            yield TransferStatus(
                status=TransferStatus.OK,
                bytes_received=file_size
            )
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"[File Service] Error in TransferShard: {e}")
            yield TransferStatus(
                status=TransferStatus.ERROR,
                error_message=str(e)
            ) 