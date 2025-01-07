import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.download.p2p.p2p_downloader import P2PShardDownloader, PeerConnectionError
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.networking.peer_handle import PeerHandle


class DownloadCoordinator(ShardDownloader):
    def __init__(
        self,
        peers: List[PeerHandle],
        disable_local_download: bool = False,
        quick_check: bool = False,
        max_parallel_downloads: int = 4
    ):
        if DEBUG >= 2:
            print(f"[Coordinator] Initializing with {len(peers)} peers, disable_local_download={disable_local_download}")
            for peer in peers:
                print(f"[Coordinator] Available peer: {peer}")
                
        self.disable_local_download = disable_local_download
        self.p2p_downloader = P2PShardDownloader(peers, quick_check) if not disable_local_download else None
        self.hf_downloader = HFShardDownloader(quick_check, max_parallel_downloads)
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()
        
        # Forward progress events from both downloaders
        if self.p2p_downloader:
            self.p2p_downloader.on_progress.add_callback(
                "coordinator",
                lambda shard, event: self._on_progress.trigger_all(shard, event)
            )
        self.hf_downloader.on_progress.add_callback(
            "coordinator",
            lambda shard, event: self._on_progress.trigger_all(shard, event)
        )

    async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
        if DEBUG >= 2:
            print(f"[Coordinator] Ensuring shard {shard} for engine {inference_engine_name}")
            print(f"[Coordinator] Local download {'disabled' if self.disable_local_download else 'enabled'}")

        if not self.disable_local_download:
            try:
                if DEBUG >= 2:
                    print(f"[Coordinator] Attempting P2P download for shard {shard}")
                return await self.p2p_downloader.ensure_shard(shard, inference_engine_name)
            except FileNotFoundError:
                if DEBUG >= 2:
                    print(f"[Coordinator] No peers have shard {shard}, falling back to HF download")
            except PeerConnectionError as e:
                if DEBUG >= 2:
                    print(f"[Coordinator] Peer connection failed for shard {shard}: {e}")
            except Exception as e:
                if DEBUG >= 2:
                    print(f"[Coordinator] P2P download failed for shard {shard}: {e}")
        else:
            if DEBUG >= 2:
                print(f"[Coordinator] P2P download disabled, using HF download for shard {shard}")
        
        # Fall back to or start with HF download
        if DEBUG >= 2:
            print(f"[Coordinator] Starting HF download for shard {shard}")
        return await self.hf_downloader.ensure_shard(shard, inference_engine_name)

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self._on_progress

    async def get_shard_download_status(self) -> Optional[Dict[str, float]]:
        # Try P2P status first if enabled
        if not self.disable_local_download:
            p2p_status = await self.p2p_downloader.get_shard_download_status()
            if p2p_status is not None:
                return p2p_status
        
        # Fall back to or use HF status
        return await self.hf_downloader.get_shard_download_status() 