from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, AsyncIterator
import numpy as np
from exo.inference.shard import Shard
from exo.topology.device_capabilities import DeviceCapabilities
from exo.topology.topology import Topology


class ModelFileInfo:
    def __init__(self, path: str, size: int, hash: str = ""):
        self.path = path
        self.size = size
        self.hash = hash


class ModelFileListResponse:
    def __init__(self, files: List[ModelFileInfo]):
        self.files = files


class FileChunk:
    def __init__(self, data: bytes, offset: int, total_size: int):
        self.data = data
        self.offset = offset
        self.total_size = total_size


class HasModelResponse:
    def __init__(self, has_model: bool, is_complete: bool, available_files: List[str] = None):
        self.has_model = has_model
        self.is_complete = is_complete
        self.available_files = available_files or []


class PeerHandle(ABC):
  @abstractmethod
  def id(self) -> str:
    pass

  @abstractmethod
  def addr(self) -> str:
    pass

  @abstractmethod
  def description(self) -> str:
    pass

  @abstractmethod
  def device_capabilities(self) -> DeviceCapabilities:
    pass

  @abstractmethod
  async def connect(self) -> None:
    pass

  @abstractmethod
  async def is_connected(self) -> bool:
    pass

  @abstractmethod
  async def disconnect(self) -> None:
    pass

  @abstractmethod
  async def health_check(self) -> bool:
    pass

  @abstractmethod
  async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_tensor(self, shard: Shard, tensor: np.array, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    pass

  @abstractmethod
  async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
    pass
    
  @abstractmethod
  async def has_model(self, repo_id: str, revision: str = "main") -> HasModelResponse:
    """Check if the peer has a given model repo downloaded"""
    pass
    
  @abstractmethod
  async def get_model_file_list(self, repo_id: str, revision: str = "main", allow_patterns: List[str] = None) -> ModelFileListResponse:
    """Get a list of files available for a model repo"""
    pass
    
  @abstractmethod
  async def get_model_file(self, repo_id: str, revision: str, file_path: str, offset: int = 0) -> AsyncIterator[FileChunk]:
    """Get a file from a model repo, streaming it in chunks"""
    pass
