import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.hf.hf_shard_download import HFShardDownloader
from exo.download.p2p.p2p_downloader import P2PShardDownloader
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
        if not self.disable_local_download:
            try:
                # First try P2P download
                if DEBUG >= 2:
                    print(f"Attempting P2P download for shard {shard}")
                return await self.p2p_downloader.ensure_shard(shard, inference_engine_name)
            except FileNotFoundError:
                # No peers have the shard, fall back to HF
                if DEBUG >= 2:
                    print(f"No peers have shard {shard}, falling back to HF download")
            except Exception as e:
                # Other P2P error, fall back to HF
                if DEBUG >= 2:
                    print(f"P2P download failed for shard {shard}: {e}")
        
        # Fall back to or start with HF download
        if DEBUG >= 2:
            print(f"Starting HF download for shard {shard}")
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