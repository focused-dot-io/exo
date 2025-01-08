from dataclasses import dataclass
from exo.networking.grpc.node_service_pb2 import Shard as ShardProto

@dataclass
class Shard:
    """Represents a model shard."""
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int

    def to_proto(self) -> ShardProto:
        """Convert to protobuf message."""
        return ShardProto(
            model_id=self.model_id,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            n_layers=self.n_layers
        )

    @classmethod
    def from_proto(cls, proto: ShardProto) -> 'Shard':
        """Create from protobuf message."""
        return cls(
            model_id=proto.model_id,
            start_layer=proto.start_layer,
            end_layer=proto.end_layer,
            n_layers=proto.n_layers
        )

    def __str__(self) -> str:
        return f"Shard(model_id='{self.model_id}', start_layer={self.start_layer}, end_layer={self.end_layer}, n_layers={self.n_layers})" 