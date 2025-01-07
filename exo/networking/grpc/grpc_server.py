import grpc
import grpc.aio as aio
from concurrent import futures
from asyncio import CancelledError
from exo import DEBUG
from exo.orchestration import Node
from .node_service_handler import NodeServiceHandler
from .file_service_handler import FileServiceHandler
from . import node_service_pb2_grpc

class GRPCServer:
    def __init__(self, node: Node, host: str, port: int):
        self.node = node
        self.host = host
        self.port = port
        self.server = None

    async def start(self) -> None:
        self.server = aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_metadata_size", 32*1024*1024),
                ("grpc.max_send_message_length", 128*1024*1024),
                ("grpc.max_receive_message_length", 128*1024*1024),
            ],
        )
        
        # Initialize service handlers
        self.node_service = NodeServiceHandler(self.node)
        self.file_service = FileServiceHandler()
        
        # Add services to server
        node_service_pb2_grpc.add_NodeServiceServicer_to_server(self.node_service, self.server)
        
        # Import file service after proto generation
        try:
            from . import file_service_pb2_grpc
            file_service_pb2_grpc.add_FileServiceServicer_to_server(self.file_service, self.server)
        except ImportError as e:
            if DEBUG >= 1:
                print(f"Warning: File service not available - {e}")
        
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

