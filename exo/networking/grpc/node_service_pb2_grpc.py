# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from . import node_service_pb2 as node__service__pb2

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
        + f' but the generated code in node_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class NodeServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendPrompt = channel.unary_unary(
                '/node_service.NodeService/SendPrompt',
                request_serializer=node__service__pb2.PromptRequest.SerializeToString,
                response_deserializer=node__service__pb2.Tensor.FromString,
                _registered_method=True)
        self.SendTensor = channel.unary_unary(
                '/node_service.NodeService/SendTensor',
                request_serializer=node__service__pb2.TensorRequest.SerializeToString,
                response_deserializer=node__service__pb2.Tensor.FromString,
                _registered_method=True)
        self.SendExample = channel.unary_unary(
                '/node_service.NodeService/SendExample',
                request_serializer=node__service__pb2.ExampleRequest.SerializeToString,
                response_deserializer=node__service__pb2.Loss.FromString,
                _registered_method=True)
        self.CollectTopology = channel.unary_unary(
                '/node_service.NodeService/CollectTopology',
                request_serializer=node__service__pb2.CollectTopologyRequest.SerializeToString,
                response_deserializer=node__service__pb2.Topology.FromString,
                _registered_method=True)
        self.SendResult = channel.unary_unary(
                '/node_service.NodeService/SendResult',
                request_serializer=node__service__pb2.SendResultRequest.SerializeToString,
                response_deserializer=node__service__pb2.Empty.FromString,
                _registered_method=True)
        self.SendOpaqueStatus = channel.unary_unary(
                '/node_service.NodeService/SendOpaqueStatus',
                request_serializer=node__service__pb2.SendOpaqueStatusRequest.SerializeToString,
                response_deserializer=node__service__pb2.Empty.FromString,
                _registered_method=True)
        self.HealthCheck = channel.unary_unary(
                '/node_service.NodeService/HealthCheck',
                request_serializer=node__service__pb2.HealthCheckRequest.SerializeToString,
                response_deserializer=node__service__pb2.HealthCheckResponse.FromString,
                _registered_method=True)
        self.GetModelFileList = channel.unary_unary(
                '/node_service.NodeService/GetModelFileList',
                request_serializer=node__service__pb2.ModelFileListRequest.SerializeToString,
                response_deserializer=node__service__pb2.ModelFileListResponse.FromString,
                _registered_method=True)
        self.GetModelFile = channel.unary_stream(
                '/node_service.NodeService/GetModelFile',
                request_serializer=node__service__pb2.ModelFileRequest.SerializeToString,
                response_deserializer=node__service__pb2.FileChunk.FromString,
                _registered_method=True)
        self.HasModel = channel.unary_unary(
                '/node_service.NodeService/HasModel',
                request_serializer=node__service__pb2.HasModelRequest.SerializeToString,
                response_deserializer=node__service__pb2.HasModelResponse.FromString,
                _registered_method=True)


class NodeServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendPrompt(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendTensor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendExample(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CollectTopology(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendResult(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendOpaqueStatus(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def HealthCheck(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelFileList(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelFile(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def HasModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NodeServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendPrompt': grpc.unary_unary_rpc_method_handler(
                    servicer.SendPrompt,
                    request_deserializer=node__service__pb2.PromptRequest.FromString,
                    response_serializer=node__service__pb2.Tensor.SerializeToString,
            ),
            'SendTensor': grpc.unary_unary_rpc_method_handler(
                    servicer.SendTensor,
                    request_deserializer=node__service__pb2.TensorRequest.FromString,
                    response_serializer=node__service__pb2.Tensor.SerializeToString,
            ),
            'SendExample': grpc.unary_unary_rpc_method_handler(
                    servicer.SendExample,
                    request_deserializer=node__service__pb2.ExampleRequest.FromString,
                    response_serializer=node__service__pb2.Loss.SerializeToString,
            ),
            'CollectTopology': grpc.unary_unary_rpc_method_handler(
                    servicer.CollectTopology,
                    request_deserializer=node__service__pb2.CollectTopologyRequest.FromString,
                    response_serializer=node__service__pb2.Topology.SerializeToString,
            ),
            'SendResult': grpc.unary_unary_rpc_method_handler(
                    servicer.SendResult,
                    request_deserializer=node__service__pb2.SendResultRequest.FromString,
                    response_serializer=node__service__pb2.Empty.SerializeToString,
            ),
            'SendOpaqueStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.SendOpaqueStatus,
                    request_deserializer=node__service__pb2.SendOpaqueStatusRequest.FromString,
                    response_serializer=node__service__pb2.Empty.SerializeToString,
            ),
            'HealthCheck': grpc.unary_unary_rpc_method_handler(
                    servicer.HealthCheck,
                    request_deserializer=node__service__pb2.HealthCheckRequest.FromString,
                    response_serializer=node__service__pb2.HealthCheckResponse.SerializeToString,
            ),
            'GetModelFileList': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModelFileList,
                    request_deserializer=node__service__pb2.ModelFileListRequest.FromString,
                    response_serializer=node__service__pb2.ModelFileListResponse.SerializeToString,
            ),
            'GetModelFile': grpc.unary_stream_rpc_method_handler(
                    servicer.GetModelFile,
                    request_deserializer=node__service__pb2.ModelFileRequest.FromString,
                    response_serializer=node__service__pb2.FileChunk.SerializeToString,
            ),
            'HasModel': grpc.unary_unary_rpc_method_handler(
                    servicer.HasModel,
                    request_deserializer=node__service__pb2.HasModelRequest.FromString,
                    response_serializer=node__service__pb2.HasModelResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'node_service.NodeService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('node_service.NodeService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class NodeService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendPrompt(request,
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
            '/node_service.NodeService/SendPrompt',
            node__service__pb2.PromptRequest.SerializeToString,
            node__service__pb2.Tensor.FromString,
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
    def SendTensor(request,
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
            '/node_service.NodeService/SendTensor',
            node__service__pb2.TensorRequest.SerializeToString,
            node__service__pb2.Tensor.FromString,
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
    def SendExample(request,
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
            '/node_service.NodeService/SendExample',
            node__service__pb2.ExampleRequest.SerializeToString,
            node__service__pb2.Loss.FromString,
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
    def CollectTopology(request,
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
            '/node_service.NodeService/CollectTopology',
            node__service__pb2.CollectTopologyRequest.SerializeToString,
            node__service__pb2.Topology.FromString,
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
    def SendResult(request,
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
            '/node_service.NodeService/SendResult',
            node__service__pb2.SendResultRequest.SerializeToString,
            node__service__pb2.Empty.FromString,
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
    def SendOpaqueStatus(request,
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
            '/node_service.NodeService/SendOpaqueStatus',
            node__service__pb2.SendOpaqueStatusRequest.SerializeToString,
            node__service__pb2.Empty.FromString,
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
    def HealthCheck(request,
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
            '/node_service.NodeService/HealthCheck',
            node__service__pb2.HealthCheckRequest.SerializeToString,
            node__service__pb2.HealthCheckResponse.FromString,
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
    def GetModelFileList(request,
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
            '/node_service.NodeService/GetModelFileList',
            node__service__pb2.ModelFileListRequest.SerializeToString,
            node__service__pb2.ModelFileListResponse.FromString,
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
    def GetModelFile(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(
            request,
            target,
            '/node_service.NodeService/GetModelFile',
            node__service__pb2.ModelFileRequest.SerializeToString,
            node__service__pb2.FileChunk.FromString,
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
    def HasModel(request,
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
            '/node_service.NodeService/HasModel',
            node__service__pb2.HasModelRequest.SerializeToString,
            node__service__pb2.HasModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
