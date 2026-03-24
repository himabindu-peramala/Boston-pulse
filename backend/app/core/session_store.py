"""
Boston Pulse — Session Store
In-memory conversation history per session.
Swap with Firestore for production.
"""
from collections import defaultdict
from typing import List
from app.core.config import settings

# session_id -> list of Gemini format messages
# {"role": "user"/"model", "parts": ["message"]}
_store: dict[str, List[dict]] = defaultdict(list)


def get_history(session_id: str) -> List[dict]:
    """Get full conversation history for a session."""
    return list(_store[session_id])


def add_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Append one user + assistant turn."""
    _store[session_id].append({"role": "user",  "parts": [user_msg]})
    _store[session_id].append({"role": "model", "parts": [assistant_msg]})
    # Keep only last N turns
    max_msgs = settings.max_history_turns * 2
    if len(_store[session_id]) > max_msgs:
        _store[session_id] = _store[session_id][-max_msgs:]


def clear_history(session_id: str) -> None:
    """Delete all history for a session."""
    _store.pop(session_id, None)
