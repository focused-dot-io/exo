import os
import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional

from exo.networking.grpc.node_service_pb2 import (
    ShardChunk, TransferStatus, Shard, ShardStatus, HealthCheckResponse
)
from exo.download.shard import Shard as ShardModel
from exo.helpers import get_local_snapshot_dir

DEBUG = int(os.environ.get("DEBUG", "0"))

class NodeServiceHandler:
    """Handler for node service requests."""

    def __init__(self, node):
        self.node = node

    async def GetShardStatus(self, request: Shard, context) -> ShardStatus:
        """Check if a shard is available locally."""
        try:
            # Get model directory
            model_dir = await get_local_snapshot_dir(request.model_id)
            if not model_dir or not model_dir.exists():
                return ShardStatus(
                    has_shard=False,
                    error_message=f"Model directory not found: {model_dir}"
                )

            # Check for model file
            model_file = model_dir / "model.safetensors"
            if not model_file.exists():
                return ShardStatus(
                    has_shard=False,
                    error_message=f"Model file not found in {model_dir}"
                )

            # Follow symlink if needed
            if model_file.is_symlink():
                real_path = model_file.resolve()
                if not real_path.exists():
                    return ShardStatus(
                        has_shard=False,
                        error_message=f"Symlink target not found: {real_path}"
                    )
                model_file = real_path

            # Check file integrity
            try:
                with open(model_file, 'rb') as f:
                    header = f.read(8)
                    if len(header) != 8:
                        return ShardStatus(
                            has_shard=False,
                            error_message="File integrity check failed"
                        )
            except Exception as e:
                return ShardStatus(
                    has_shard=False,
                    error_message=f"Error reading file: {e}"
                )

            return ShardStatus(has_shard=True)

        except Exception as e:
            return ShardStatus(
                has_shard=False,
                error_message=str(e)
            )

    async def TransferShard(self, request_iterator: AsyncIterator[ShardChunk], context) -> AsyncIterator[TransferStatus]:
        """Handle shard transfer requests from peers."""
        try:
            # Get initial request with metadata
            initial_request = await anext(request_iterator)
            if not initial_request.HasField('metadata'):
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message="First message must contain metadata"
                )
                return

            # Extract shard info
            shard = ShardModel.from_proto(initial_request.metadata.shard)
            inference_engine = initial_request.metadata.inference_engine

            # Get model directory
            model_dir = await get_local_snapshot_dir(shard.model_id)
            if not model_dir or not model_dir.exists():
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message=f"Model directory not found: {model_dir}"
                )
                return

            # Verify model directory structure
            if not (model_dir / "model.safetensors").exists():
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message=f"Model file not found in {model_dir}"
                )
                return

            # Get list of files to transfer
            files_to_transfer = []
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    # Follow symlinks
                    if file_path.is_symlink():
                        real_path = file_path.resolve()
                        if not real_path.exists():
                            continue
                        file_path = real_path
                    
                    # Get relative path from model directory
                    rel_path = file_path.relative_to(model_dir)
                    files_to_transfer.append((file_path, rel_path))

            if not files_to_transfer:
                yield TransferStatus(
                    status=TransferStatus.ERROR,
                    error_message=f"No files found in {model_dir}"
                )
                return

            # Calculate total size
            total_size = sum(file_path.stat().st_size for file_path, _ in files_to_transfer)

            # Send initial status with total size
            yield TransferStatus(
                status=TransferStatus.OK,
                bytes_received=0,
                total_bytes=total_size
            )

            bytes_sent = 0
            chunk_size = 1024 * 1024  # 1MB chunks

            # Transfer each file
            for file_path, rel_path in files_to_transfer:
                # Send file metadata
                yield ShardChunk(
                    metadata=ShardChunk.Metadata(
                        file_name=str(rel_path),
                        file_size=file_path.stat().st_size
                    )
                )

                # Transfer file in chunks
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break

                        # Send chunk
                        yield ShardChunk(
                            chunk_data=chunk,
                            offset=bytes_sent
                        )
                        bytes_sent += len(chunk)

                        # Wait for acknowledgment
                        try:
                            ack = await anext(request_iterator)
                            if isinstance(ack, TransferStatus):
                                if ack.status == TransferStatus.ERROR:
                                    error_msg = getattr(ack, 'error_message', 'Unknown error')
                                    yield TransferStatus(
                                        status=TransferStatus.ERROR,
                                        error_message=f"Client error: {error_msg}"
                                    )
                                    return
                        except StopAsyncIteration:
                            yield TransferStatus(
                                status=TransferStatus.ERROR,
                                error_message="Client disconnected"
                            )
                            return

                # Send end of file marker
                yield ShardChunk(is_last=True)

            # Send final status
            yield TransferStatus(
                status=TransferStatus.OK,
                bytes_received=total_size,
                total_bytes=total_size
            )

        except Exception as e:
            yield TransferStatus(
                status=TransferStatus.ERROR,
                error_message=str(e)
            )

    async def HealthCheck(self, request, context) -> HealthCheckResponse:
        """Handle health check requests."""
        try:
            return HealthCheckResponse(is_healthy=True)
        except Exception as e:
            if DEBUG >= 2:
                print(f"[Node Service] Health check failed: {e}")
            return HealthCheckResponse(is_healthy=False) 