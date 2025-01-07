# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from exo.networking.grpc import file_service_pb2 as exo_dot_networking_dot_grpc_dot_file__service__pb2

GRPC_GENERATED_VERSION = '1.68.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in exo/networking/grpc/file_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class FileServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetShardStatus = channel.unary_unary(
                '/file_service.FileService/GetShardStatus',
                request_serializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.GetShardStatusRequest.SerializeToString,
                response_deserializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.GetShardStatusResponse.FromString,
                _registered_method=True)
        self.TransferShard = channel.stream_stream(
                '/file_service.FileService/TransferShard',
                request_serializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.ShardChunk.SerializeToString,
                response_deserializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.TransferStatus.FromString,
                _registered_method=True)


class FileServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetShardStatus(self, request, context):
        """Check if a peer has a specific shard
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TransferShard(self, request_iterator, context):
        """Transfer a shard file from one peer to another
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FileServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetShardStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.GetShardStatus,
                    request_deserializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.GetShardStatusRequest.FromString,
                    response_serializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.GetShardStatusResponse.SerializeToString,
            ),
            'TransferShard': grpc.stream_stream_rpc_method_handler(
                    servicer.TransferShard,
                    request_deserializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.ShardChunk.FromString,
                    response_serializer=exo_dot_networking_dot_grpc_dot_file__service__pb2.TransferStatus.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'file_service.FileService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('file_service.FileService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class FileService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetShardStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/file_service.FileService/GetShardStatus',
            exo_dot_networking_dot_grpc_dot_file__service__pb2.GetShardStatusRequest.SerializeToString,
            exo_dot_networking_dot_grpc_dot_file__service__pb2.GetShardStatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def TransferShard(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(
            request_iterator,
            target,
            '/file_service.FileService/TransferShard',
            exo_dot_networking_dot_grpc_dot_file__service__pb2.ShardChunk.SerializeToString,
            exo_dot_networking_dot_grpc_dot_file__service__pb2.TransferStatus.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
