"""Pydantic data models for Mnemosyne."""

import json

from pydantic import BaseModel, Field


class Peer(BaseModel):
    """A user or agent that interacts with the memory system."""

    id: str
    name: str
    peer_type: str = "user"
    static_profile: dict | None = None
    profile_updated_at: str | None = None
    created_at: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict) -> "Peer":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("static_profile"), str):
            data["static_profile"] = json.loads(data["static_profile"])
        elif data.get("static_profile") is None:
            data["static_profile"] = None
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


class Session(BaseModel):
    """A conversation session with a peer."""

    id: str
    peer_id: str
    started_at: str
    ended_at: str | None = None
    summary: str | None = None
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict) -> "Session":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


class Message(BaseModel):
    """A single message within a session."""

    id: str
    session_id: str
    peer_id: str
    role: str
    content: str
    created_at: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict) -> "Message":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


class Note(BaseModel):
    """A memory note — the core unit of the memory system."""

    id: str
    peer_id: str
    session_id: str | None = None
    source_message_id: str | None = None
    content: str
    context_description: str | None = None
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    note_type: str = "observation"
    provenance: str = "organic"
    durability: str = "contextual"
    emotional_weight: float = 0.5
    importance: float = 0.0
    confidence: float = 0.8
    evidence_count: int = 1
    unique_sessions_mentioned: int = 1
    q_value: float = 0.0
    access_count: int = 0
    last_accessed_at: str | None = None
    times_surfaced: int = 0
    decay_score: float = 1.0
    is_buffered: bool = True
    canonical_note_id: str | None = None
    created_at: str
    updated_at: str
    zvec_id: str | None = None

    @classmethod
    def from_row(cls, row: dict) -> "Note":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("keywords"), str):
            data["keywords"] = json.loads(data["keywords"])
        if isinstance(data.get("tags"), str):
            data["tags"] = json.loads(data["tags"])
        # SQLite stores bools as 0/1
        if "is_buffered" in data:
            data["is_buffered"] = bool(data["is_buffered"])
        return cls(**data)


class Link(BaseModel):
    """A typed relationship between two notes."""

    id: str
    source_note_id: str
    target_note_id: str
    link_type: str
    strength: float = 0.5
    created_at: str
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict) -> "Link":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**data)


class PeerProfile(BaseModel):
    """A static profile (Peer Card) generated from permanent notes."""

    peer_id: str
    sections: dict[str, str]
    fact_count: int
    generated_at: str
    source_note_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict) -> "PeerProfile":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("sections"), str):
            data["sections"] = json.loads(data["sections"])
        if isinstance(data.get("source_note_ids"), str):
            data["source_note_ids"] = json.loads(data["source_note_ids"])
        return cls(**data)


class TaskItem(BaseModel):
    """A task in the background processing queue."""

    id: str
    task_type: str
    payload: dict = Field(default_factory=dict)
    status: str = "pending"
    priority: int = 0
    attempts: int = 0
    max_attempts: int = 3
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None

    @classmethod
    def from_row(cls, row: dict) -> "TaskItem":
        """Construct from an aiosqlite Row dict."""
        data = dict(row)
        if isinstance(data.get("payload"), str):
            data["payload"] = json.loads(data["payload"])
        return cls(**data)


class RetrievalResult(BaseModel):
    """A scored retrieval result with full scoring breakdown."""

    note: Note
    score: float
    composite_score: float = 0.0
    rrf_score: float
    decay_strength: float
    provenance_weight: float
    fatigue_factor: float
    inference_discount: float
    colbert_score: float | None = None
    source: str
