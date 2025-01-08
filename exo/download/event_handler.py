from typing import Callable, List, Dict, TypeVar, Generic, Any
from asyncio import iscoroutinefunction

K = TypeVar('K')
V = TypeVar('V')

class Callback:
    """Represents a single callback registration."""
    
    def __init__(self, event_handler: 'EventHandler', key: K):
        self._event_handler = event_handler
        self._key = key
        self._callbacks: List[Callable[..., Any]] = []

    def on_next(self, callback: Callable[..., Any]) -> None:
        """Add a callback to be triggered on next event."""
        self._event_handler.add_callback(self._key, callback)

class EventHandler(Generic[K, V]):
    """Simple event handler for managing callbacks."""
    
    def __init__(self):
        self._callbacks: Dict[K, List[Callable[..., Any]]] = {}

    def register(self, key: K) -> Callback:
        """Register a new event key and return a callback object."""
        if key not in self._callbacks:
            self._callbacks[key] = []
        return Callback(self, key)

    def add_callback(self, key: K, callback: Callable[..., Any]) -> None:
        """Add a callback for a specific event key."""
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def remove_callback(self, key: K, callback: Callable[..., Any]) -> None:
        """Remove a callback for a specific event key."""
        if key in self._callbacks and callback in self._callbacks[key]:
            self._callbacks[key].remove(callback)

    async def trigger_all(self, key: K, *args: Any, **kwargs: Any) -> None:
        """Trigger all callbacks for a specific event key."""
        if key in self._callbacks:
            for callback in self._callbacks[key]:
                if iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs) 