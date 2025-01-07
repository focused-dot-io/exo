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
        self.is_running = False

    async def start(self) -> None:
        if self.is_running:
            if DEBUG >= 2:
                print(f"[GRPC Server] Server already running on {self.host}:{self.port}")
            return

        self.server = aio.server(
            futures.ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_metadata_size", 32*1024*1024),
                ("grpc.max_send_message_length", 128*1024*1024),
                ("grpc.max_receive_message_length", 128*1024*1024),
                ("grpc.keepalive_time_ms", 10000),  # Send keepalive every 10 seconds
                ("grpc.keepalive_timeout_ms", 5000),  # 5 second timeout for keepalive
                ("grpc.keepalive_permit_without_calls", True),  # Allow keepalive without active calls
                ("grpc.http2.max_pings_without_data", 0),  # Allow unlimited pings
                ("grpc.http2.min_time_between_pings_ms", 10000),  # Minimum 10s between pings
                ("grpc.http2.min_ping_interval_without_data_ms", 5000),  # Minimum 5s between pings when idle
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
            if DEBUG >= 2:
                print("[GRPC Server] File service registered")
        except ImportError as e:
            if DEBUG >= 1:
                print(f"Warning: File service not available - {e}")
        
        listen_addr = f"{self.host}:{self.port}"
        try:
            self.server.add_insecure_port(listen_addr)
            await self.server.start()
            self.is_running = True
            if DEBUG >= 1:
                print(f"[GRPC Server] Server started, listening on {listen_addr}")
        except Exception as e:
            if DEBUG >= 1:
                print(f"[GRPC Server] Failed to start server on {listen_addr}: {e}")
            raise

    async def stop(self) -> None:
        if self.server and self.is_running:
            try:
                self.is_running = False
                await self.server.stop(grace=5)
                await self.server.wait_for_termination()
                if DEBUG >= 1:
                    print("[GRPC Server] Server stopped and all connections are closed")
            except CancelledError:
                if DEBUG >= 2:
                    print("[GRPC Server] Server stop cancelled")
                pass
            except Exception as e:
                if DEBUG >= 1:
                    print(f"[GRPC Server] Error stopping server: {e}")
                raise

