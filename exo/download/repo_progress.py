from dataclasses import dataclass
from datetime import timedelta
from typing import Dict

@dataclass
class RepoProgressEvent:
    """Event for tracking repository download progress."""
    repo_id: str
    repo_revision: str
    completed_files: int
    total_files: int
    downloaded_bytes: int
    downloaded_bytes_this_session: int
    total_bytes: int
    overall_speed: float
    overall_eta: timedelta
    file_progress: Dict[str, float]
    status: str 