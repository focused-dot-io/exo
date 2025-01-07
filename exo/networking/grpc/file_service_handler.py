import asyncio
from pathlib import Path
from typing import Optional, AsyncIterator
from exo.inference.shard import Shard
from exo.networking.grpc.file_service_pb2 import (
    GetShardStatusRequest, GetShardStatusResponse,
    ShardChunk, TransferStatus
)
from exo.networking.grpc.file_service_pb2_grpc import FileServiceServicer
from exo.download.hf.hf_helpers import get_local_snapshot_dir
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
            shard = Shard.from_proto(request.shard)
            repo_name = get_repo(shard.model_id, request.inference_engine_name)
            
            # Check if we have the shard locally
            snapshot_dir = await get_local_snapshot_dir(repo_name)
            if not snapshot_dir:
                return GetShardStatusResponse(has_shard=False)
                
            # TODO: Add logic to find exact shard file path
            # For now just report we have it if we have the snapshot
            return GetShardStatusResponse(
                has_shard=True,
                local_path=str(snapshot_dir),
                file_size=0  # TODO: Get actual file size
            )
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"Error in GetShardStatus: {e}")
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
            shard = Shard.from_proto(metadata.shard)
            repo_name = get_repo(shard.model_id, metadata.inference_engine_name)
            
            # Find shard file
            snapshot_dir = await get_local_snapshot_dir(repo_name)
            if not snapshot_dir:
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message="Shard not found locally"
                )
                return
                
            # TODO: Add logic to find exact shard file path
            # For now just use snapshot dir
            file_path = snapshot_dir
            
            # Send file in chunks
            file_size = file_path.stat().st_size
            bytes_sent = 0
            
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                        
                    bytes_sent += len(chunk)
                    yield TransferStatus(
                        status=TransferStatus.OK,
                        bytes_received=bytes_sent
                    )
                    
                    await asyncio.sleep(0)  # Yield control
                    
            # Send final status
            yield TransferStatus(
                status=TransferStatus.OK,
                bytes_received=file_size
            )
            
        except Exception as e:
            if DEBUG >= 2:
                print(f"Error in TransferShard: {e}")
            yield TransferStatus(
                status=TransferStatus.ERROR,
                error_message=str(e)
            ) 