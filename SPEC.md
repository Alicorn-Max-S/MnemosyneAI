# Mnemosyne — Spec

AI agent memory system. SQLite + FTS5 + Zvec for storage/search. Async write pipeline (Deriver via DeepSeek V3.2). Retrieval pipeline (parallel search, RRF fusion, scoring, MMR dedup). Intelligence layer (A-MEM links, ColBERT reranking, static profile). Background processes (batch dedup, Dreamer via Gemini 3 Flash Batch API, MAGMA multi-graph). Refinement layer (MemR3 reflective loop, MemRL Q-value tracking, anti-sycophancy guards, adaptive routing, user feedback and memory correction).

---

## Project Structure

```
mnemosyne/
├── pyproject.toml
├── CLAUDE.md
├── SPEC.md                     ← this file
├── mnemosyne/
│   ├── __init__.py
│   ├── config.py               # All constants, thresholds, API config
│   ├── models.py               # Pydantic: Peer, Session, Message, Note, Link, TaskItem, RetrievalResult, ReflectiveResult, FeedbackEvent
│   ├── utils/
│   │   └── ids.py              # ULID generation
│   ├── db/
│   │   └── sqlite_store.py     # All SQLite CRUD, FTS5 search, task queue, access tracking
│   ├── vectors/
│   │   ├── zvec_store.py       # Zvec insert/search/delete wrapper
│   │   └── embedder.py         # nomic-embed-text-v1.5 (ONNX preferred, PyTorch fallback)
│   ├── api/
│   │   └── memory_api.py       # Public API coordinating all stores
│   ├── pipeline/               # Write path
│   │   ├── __init__.py         # create_worker() factory
│   │   ├── intake.py           # Fast-path message ingestion (<5ms)
│   │   ├── deriver.py          # Extractor + Scorer (DeepSeek V3.2 via NousResearch)
│   │   ├── handlers.py         # handle_derive: extract → score → embed → store
│   │   └── worker.py           # Queue polling loop + dispatch
│   ├── retrieval/              # Read path
│   │   ├── __init__.py         # create_retriever() factory
│   │   ├── scorer.py           # Pure functions: decay, provenance weight, fatigue, composite
│   │   ├── fusion.py           # RRF fusion + MMR dedup
│   │   └── retriever.py        # Orchestrator: parallel search → fuse → score → dedup → return
│   ├── intelligence/           # A-MEM links, ColBERT, static profile
│   │   ├── __init__.py
│   │   ├── link_generator.py   # Cosine-similarity link creation (threshold 0.75)
│   │   ├── colbert_reranker.py # answerai-colbert-small-v1 via pylate (pre-computed tokens)
│   │   └── profile_generator.py # Static Peer Card generation via Deriver API
│   ├── dreamer/                # Background processing
│   │   ├── __init__.py         # create_dreamer() factory
│   │   ├── dedup.py            # Batch dedup: cosine clustering + union-find merge
│   │   ├── gemini_client.py    # Gemini Batch API wrapper (google-genai SDK)
│   │   ├── prompts.py          # System prompts for all Dreamer tasks
│   │   ├── task_builder.py     # Builds batch request payloads from buffered notes
│   │   ├── processor.py        # Processes batch results back into stores
│   │   └── orchestrator.py     # Coordinates full Dreamer cycle
│   ├── graph/                  # MAGMA multi-graph
│   │   ├── __init__.py
│   │   └── magma.py            # Entity extraction, graph operations, link expansion
│   └── refinement/             # Adaptive retrieval, MemR3, anti-sycophancy
│       ├── __init__.py         # create_router(), create_reflective_retriever()
│       ├── router.py           # Simple vs complex query classification
│       ├── reflective.py       # MemR3 evidence-gap tracker + retrieve/reflect/answer loop
│       ├── qvalue.py           # MemRL Q-value tracking + Bellman updates
│       ├── feedback.py         # User feedback ingestion (explicit + implicit signals)
│       ├── antisycophancy.py   # Epsilon-greedy exploration, topic cluster cap, prompt injection
│       └── correction.py       # Memory correction flow (invalidate + replace)
├── tests/
│   ├── conftest.py
│   ├── test_sqlite_store.py
│   ├── test_embedder.py
│   ├── test_zvec_store.py
│   ├── test_memory_api.py
│   ├── test_worker.py
│   ├── test_intake.py
│   ├── test_deriver.py
│   ├── test_write_pipeline.py
│   ├── test_scorer.py
│   ├── test_fusion.py
│   ├── test_retriever.py
│   ├── test_retrieval_pipeline.py
│   ├── test_reranker.py
│   ├── test_linker.py
│   ├── test_profiler.py
│   ├── test_link_expansion.py
│   ├── test_intelligence_pipeline.py
│   ├── test_dedup.py
│   ├── test_gemini_client.py
│   ├── test_dreamer_orchestrator.py
│   ├── test_magma.py
│   ├── test_dreamer_pipeline.py
│   ├── test_qvalue.py
│   ├── test_reflective.py
│   ├── test_router.py
│   ├── test_feedback.py
│   ├── test_correction.py
│   ├── test_antisycophancy.py
│   └── test_refinement_pipeline.py
```

No FastAPI, no Celery, no Redis, no Postgres, no Docker Compose needed for the core system. Those are not needed.

---

## config.py

All constants live here.

```python
# Storage
DEFAULT_DATA_DIR = "./mnemosyne_data"
SQLITE_DB_NAME = "mnemosyne.db"
ZVEC_COLLECTION_DIR = "zvec_notes"
ZVEC_COLLECTION_NAME = "mnemosyne_notes"

# Embedding
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 384
EMBEDDING_QUERY_PREFIX = "search_query: "
EMBEDDING_DOC_PREFIX = "search_document: "

# Schema
SCHEMA_VERSION = 1
RRF_K = 60

# Provenance values
PROVENANCE_ORGANIC = "organic"
PROVENANCE_AGENT_PROMPTED = "agent_prompted"
PROVENANCE_USER_CONFIRMED = "user_confirmed"
PROVENANCE_INFERRED = "inferred"

# Note types / durability
NOTE_TYPE_OBSERVATION = "observation"
NOTE_TYPE_INFERENCE = "inference"
DURABILITY_PERMANENT = "permanent"
DURABILITY_CONTEXTUAL = "contextual"
DURABILITY_EPHEMERAL = "ephemeral"

# Link types
LINK_TYPES = ["semantic", "causal", "temporal", "contradicts", "supports", "derived_from"]

# Defaults
DEFAULT_CONFIDENCE_OBSERVATION = 0.8
DEFAULT_CONFIDENCE_INFERENCE = 0.6
DEFAULT_CONFIDENCE_USER_SET = 1.0
DEFAULT_EMOTIONAL_WEIGHT = 0.5

# Deriver API
NOUSRESEARCH_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUSRESEARCH_MODEL = "deepseek/deepseek-v3.2"
DERIVER_EXTRACT_TEMPERATURE = 0.1
DERIVER_SCORE_TEMPERATURE = 0.1
DERIVER_MAX_RETRIES = 3
DERIVER_RETRY_DELAYS = [1.0, 2.0, 4.0]
WORKER_POLL_INTERVAL = 2.0

# Retrieval scoring
PROVENANCE_WEIGHTS = {"organic": 1.0, "user_confirmed": 0.8, "agent_prompted": 0.5, "inferred": 0.3}
DECAY_BASE_LAMBDA = 0.1
DECAY_IMPORTANCE_FACTOR = 0.8
DECAY_ACCESS_BOOST = 0.15
DECAY_MIN_MEMORIES = 100
DECAY_RAMP_MAX = 1000
DECAY_HIGH_IMPORTANCE_FLOOR = 0.3
SURFACING_FATIGUE_FACTOR = 0.1
MMR_SIMILARITY_THRESHOLD = 0.90
INFERENCE_SCORE_DISCOUNT = 0.7
RETRIEVAL_FTS_LIMIT = 30
RETRIEVAL_VECTOR_LIMIT = 30
RETRIEVAL_FINAL_LIMIT = 10

# A-MEM link generation
LINK_SIMILARITY_THRESHOLD = 0.75
LINK_CANDIDATE_LIMIT = 20
LINK_MAX_PER_NOTE = 5

# ColBERT reranking (pre-computed token embeddings)
COLBERT_MODEL = "answerdotai/answerai-colbert-small-v1"
COLBERT_TOKEN_DIM = 96
COLBERT_RERANK_LIMIT = 50
COLBERT_TOP_N = 10

# Static profile
PROFILE_MAX_FACTS = 30
PROFILE_SECTIONS = ["identity", "professional", "communication_style", "relationships"]
PROFILE_GENERATION_TEMPERATURE = 0.1

# Batch dedup
DEDUP_COSINE_THRESHOLD = 0.85
DEDUP_MIN_CLUSTER_SIZE = 2

# Dreamer — Gemini Batch API
GEMINI_MODEL = "gemini-3-flash-preview"
DREAMER_TEMPERATURE = 0.3
DREAMER_POLL_INTERVAL = 30.0
DREAMER_MAX_POLL_TIME = 86400  # 24 hours
DREAMER_LINK_TEMPERATURE = 0.3
DREAMER_PATTERN_TEMPERATURE = 0.3
DREAMER_CONTRADICTION_TEMPERATURE = 0.1
DREAMER_PROFILE_TEMPERATURE = 0.1

# MAGMA graph
ENTITY_GRAPH_NAME = "entity"
TEMPORAL_GRAPH_NAME = "temporal"
CAUSAL_GRAPH_NAME = "causal"

# MemRL Q-value tracking
QVALUE_DEFAULT = 0.0
QVALUE_LEARNING_RATE = 0.1
QVALUE_POSITIVE_REWARD = 1.0
QVALUE_NEGATIVE_REWARD = -0.5
QVALUE_IMPLICIT_POSITIVE = 0.05
QVALUE_IMPLICIT_NEGATIVE = -0.05
QVALUE_WEIGHT_IN_COMPOSITE = 0.15
QVALUE_ZSCORE_WINDOW = 100
QVALUE_ZSCORE_MIN_SAMPLES = 10

# Adaptive retrieval routing
ROUTER_SIMPLE_THRESHOLD = 0.6
ROUTER_COMPLEX_BUDGET = 3
ROUTER_DISTANCE_PERCENTILE = 75
ROUTER_MIN_CANDIDATES = 3

# MemR3 reflective loop
MEMR3_MAX_ITERATIONS = 3
MEMR3_EVIDENCE_CONFIDENCE_THRESHOLD = 0.7
MEMR3_MASK_THRESHOLD = 0.85
MEMR3_REFLECT_MODEL = "deepseek/deepseek-v3.2"
MEMR3_REFLECT_TEMPERATURE = 0.1
MEMR3_REFLECT_MAX_RETRIES = 2

# Anti-sycophancy
ANTISYC_TOPIC_CLUSTER_CAP = 0.4
ANTISYC_EPSILON = 0.1
ANTISYC_AGENT_TOPIC_DECAY_SESSIONS = 3
ANTISYC_CLUSTER_SIMILARITY = 0.75

# Feedback
FEEDBACK_CORRECTION_CONFIDENCE = 0.3
FEEDBACK_REPLACEMENT_PROVENANCE = "user_confirmed"
FEEDBACK_SESSION_WINDOW = 10
```

---

## SQLite Schema

PRAGMAs applied on every connection:
```sql
PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-8000;
PRAGMA busy_timeout=5000; PRAGMA mmap_size=268435456; PRAGMA temp_store=MEMORY;
PRAGMA foreign_keys=ON;
```

All PKs are TEXT (ULIDs). Timestamps: `strftime('%Y-%m-%dT%H:%M:%fZ', 'now')`.

**Tables:** config, peers, sessions, messages, notes, links, task_queue, peer_profiles, entity_mentions, colbert_tokens, feedback_events, agent_topic_tracker. See `sqlite_store.py` for full DDL. The critical table is **notes**:

| Column group | Columns |
|---|---|
| Identity | id, peer_id, session_id, source_message_id |
| Content | content, context_description, keywords (JSON), tags (JSON) |
| Classification | note_type, provenance, durability |
| Scoring | emotional_weight, importance, confidence |
| Frequency | evidence_count, unique_sessions_mentioned |
| Retrieval state | q_value, access_count, last_accessed_at, times_surfaced, decay_score |
| Pipeline state | is_buffered, canonical_note_id, zvec_id |
| Correction state | is_invalidated, invalidated_by, correction_of |
| Timestamps | created_at, updated_at |

**entity_mentions** — tracks entities extracted from notes for the MAGMA entity graph:

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | ULID |
| note_id | TEXT FK | References notes(id) |
| peer_id | TEXT FK | References peers(id) |
| entity_name | TEXT | Normalized entity name (lowercased) |
| entity_type | TEXT | person, place, organization, concept, other |
| mention_context | TEXT | Sentence or phrase containing the mention |
| created_at | TEXT | Timestamp |

**colbert_tokens** — pre-computed ColBERT token-level embeddings for MaxSim reranking:

| Column | Type | Description |
|---|---|---|
| note_id | TEXT PK FK | References notes(id), one row per note |
| token_embeddings | BLOB | Serialized numpy array of shape (num_tokens, 96) |
| num_tokens | INTEGER | Number of tokens (for quick size checks without deserializing) |
| created_at | TEXT | Timestamp |

**feedback_events** — tracks all user feedback signals:

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | ULID |
| peer_id | TEXT FK | References peers(id) |
| session_id | TEXT FK | References sessions(id), nullable |
| note_ids | TEXT | JSON array of note IDs that were surfaced |
| feedback_type | TEXT | "thumbs_up", "thumbs_down", "correction", "implicit_positive", "implicit_negative" |
| correction_text | TEXT | For corrections: the replacement content, nullable |
| created_at | TEXT | Timestamp |

**agent_topic_tracker** — detects agent-introduced topics persisting without organic user mention:

| Column | Type | Description |
|---|---|---|
| id | TEXT PK | ULID |
| peer_id | TEXT FK | References peers(id) |
| topic_cluster | TEXT | Representative content or embedding ID |
| consecutive_agent_sessions | INTEGER | Count of sessions with agent mention but no organic mention |
| last_organic_session_id | TEXT | Session where user last mentioned this topic organically |
| last_seen_session_id | TEXT | Most recent session where this topic appeared |
| flagged_for_decay | INTEGER | 1 if consecutive_agent_sessions >= 3 |
| created_at | TEXT | Timestamp |
| updated_at | TEXT | Timestamp |

**FTS5** virtual table on (content, context_description, keywords, tags) with sync triggers for INSERT/UPDATE/DELETE. Without triggers, FTS5 returns nothing.

---

## SQLiteStore Methods

**CRUD**: peers, sessions, messages, notes, links. Task queue with atomic dequeue (UPDATE...RETURNING). `fts_search()` with BM25 scoring.

**Retrieval support:**

- `fts_search_ranked(query, peer_id, limit=30)` → note dicts with 0-indexed `fts_rank` positions. RRF needs rank positions, not raw scores.
- `get_notes_by_ids(note_ids)` → fetch full rows for a list of IDs. Used to hydrate Zvec results from SQLite.
- `record_access(note_ids)` → batch UPDATE incrementing access_count, times_surfaced, and setting last_accessed_at. Single transaction.
- `count_notes(peer_id)` → integer count for decay scoring.

**Link expansion:**

- `get_linked_note_ids(note_id, max_depth=1)` → BFS walk returning connected note IDs.
- `get_links_by_type(note_id, link_type)` → filter links by type.

**Batch dedup:**

- `get_buffered_notes(peer_id)` → all notes with `is_buffered=1` for a peer, ordered by created_at.
- `merge_notes(canonical_id, merged_ids)` → set `canonical_note_id` on merged notes, sum `evidence_count` on canonical, set `is_buffered=0` on all. Single transaction.
- `get_unique_sessions_for_notes(note_ids)` → count distinct session_ids across a set of notes.

**Entity graph:**

- `add_entity_mention(note_id, peer_id, entity_name, entity_type, mention_context)` → insert into entity_mentions.
- `get_entity_mentions(peer_id, entity_name)` → all mentions of an entity for a peer.
- `get_entities_for_peer(peer_id)` → distinct entity names and types.

**ColBERT token storage:**

- `store_colbert_tokens(note_id, token_embeddings_blob, num_tokens)` → insert or replace into colbert_tokens.
- `get_colbert_tokens(note_ids) → dict[str, bytes]` — batch fetch token embedding BLOBs for a list of note IDs. Returns {note_id: blob} dict. Missing IDs are omitted.
- Token embeddings are stored as `numpy.ndarray.tobytes()` and reconstructed with `numpy.frombuffer(...).reshape(num_tokens, COLBERT_TOKEN_DIM)`.

**Q-value and feedback:**

- `update_qvalue(note_id, q_value)` → update q_value column on a single note.
- `batch_update_qvalues(updates: dict[str, float])` → update q_value for multiple notes. Single transaction.
- `get_qvalues(peer_id, limit=100)` → return recent notes with their q_values, ordered by created_at desc.
- `create_feedback_event(id, peer_id, session_id, note_ids, feedback_type, correction_text)` → insert into feedback_events.
- `get_feedback_events(peer_id, limit=50)` → return recent feedback events.
- `invalidate_note(note_id, invalidated_by)` → set `is_invalidated=1`, `invalidated_by=<replacement_id>` on the note.
- `get_notes_by_qvalue(peer_id, limit)` → notes sorted by q_value descending.

---

## Embedder / ZvecStore / MemoryAPI

Already implemented. See source files for details.

**Key facts:**
- `Embedder.embed_query(text)` prepends `"search_query: "`, returns 384-dim L2-normalized list[float].
- `Embedder.embed_document(text)` prepends `"search_document: "`, returns 384-dim L2-normalized list[float].
- `Embedder.embed_documents(texts)` batch version for MMR embedding.
- `ZvecStore.search(query_embedding, top_k)` returns `[{"id": str, "score": float}, ...]`.
- Zvec has **no scalar fields/filtering** — returns results across all peers. Post-filter by peer_id after hydrating from SQLite.
- MemoryAPI gains `retrieve(query, peer_id, limit)` which delegates to the Retriever, `retrieve_reflective(query, peer_id, limit)` which delegates to the ReflectiveRetriever, `run_dreamer_cycle(peer_id)` which delegates to the Dreamer orchestrator, `record_feedback(peer_id, session_id, note_ids, feedback_type, correction_text)` which delegates to FeedbackProcessor, and `get_feedback_history(peer_id, limit)` which delegates to FeedbackProcessor.

---

## Write Pipeline

1. **intake.py**: `ingest_message()` writes to SQLite + enqueues "derive" task for user messages. Assistant messages stored but not derived. < 5ms.
2. **deriver.py**: `Deriver.extract()` → atomic facts from user message. `Deriver.score()` → tags each note with emotional_weight, provenance, durability, keywords, tags, context_description. Both call DeepSeek V3.2 via NousResearch (httpx, not openai SDK). Retry 3x with 1/2/4s backoff.
3. **handlers.py**: `handle_derive()` wires extract → score → embed → store (SQLite + Zvec). Also pre-computes ColBERT token embeddings and stores them, and extracts entities for MAGMA.
4. **worker.py**: polls task queue, dispatches to handlers, manages retries/dead-letter.

**Critical rules**: Agent responses are read for context but NEVER extracted from. User confirmations get `provenance: "user_confirmed"`. Notes stored with `is_buffered=1` for the Dreamer.

---

## Retrieval Pipeline

No LLM calls in the standard path. Pure local computation. Target: < 100ms with ColBERT, < 80ms without. Complex queries via MemR3 reflective loop: 2-4s (1-3 LLM calls).

```
Query
  ├→ embed_query() ──→ Zvec search (top 30)
  ├→ FTS5 search (top 30)
  ├→ Link expansion on top vector hits (BFS depth 1)
  │         ↓
  │    Hydrate from SQLite + peer_id filter
  │    Exclude invalidated notes (is_invalidated=1)
  │         ↓
  │    RRF Fusion (k=60) across all three streams
  │         ↓
  │    Score multipliers: decay × provenance × fatigue × inference_discount × agent_topic_penalty + Q-value boost
  │         ↓
  │    ColBERT rerank top 50 → top 10
  │    (encode query tokens, load pre-computed doc tokens from SQLite, MaxSim)
  │         ↓
  │    MMR Dedup (cosine > 0.90 → drop lower-scored)
  │         ↓
  │    Anti-sycophancy: topic cluster diversity cap (40%)
  │         ↓
  │    Anti-sycophancy: epsilon-greedy exploration (p=0.1)
  │         ↓
  │    Router: classify as simple or complex
  │         ↓
  │    [If complex] MemR3 reflective loop (1-3 iterations)
  │         ↓
  │    Top-N (default 10) → record_access() → return
```

### scorer.py — Pure scoring functions

All functions are stateless, no I/O.

- `compute_decay_strength(importance, days_since_access, access_count, total_memories)` — Ebbinghaus decay with importance-weighted lambda, access boost, ramp from 100–1000 memories, floor of 0.3 for high importance.
- `compute_provenance_weight(provenance)` — Lookup from `PROVENANCE_WEIGHTS`.
- `compute_surfacing_fatigue(times_surfaced)` — `1 / (1 + 0.1 * times_surfaced)`
- `compute_inference_discount(note_type)` — "inference" → 0.7, else → 1.0.
- `compute_agent_topic_penalty(note_id, flagged_ids)` — 0.5 if the note is in `flagged_ids` (halves its score), 1.0 otherwise.
- `compute_composite_score(rrf_score, decay, provenance, fatigue, inference_discount, q_boost=0.0, agent_topic_penalty=1.0)` — `base = rrf × decay × provenance × fatigue × inference_discount × agent_topic_penalty; return base + q_boost`. Q-value is additive to prevent a single bad Q from zeroing out a relevant memory.

### fusion.py — RRF + MMR

- `rrf_fuse(ranked_lists, k=60)` — 0-indexed ranks. Notes in multiple lists get multiple contributions.
- `mmr_dedup(scored_ids, embeddings, threshold=0.90)` — cosine = dot product for L2-normalized vectors. Missing embeddings → auto-accept.

### retriever.py — Orchestrator

Steps: parallel search → hydrate → exclude invalidated → RRF → score (with Q-value boost and agent topic penalty) → ColBERT rerank (query encode + pre-computed MaxSim) → MMR dedup → topic diversity cap → epsilon exploration → truncate → record access → return.

**Error handling**: Zvec fails → FTS-only. FTS fails → vector-only. ColBERT fails → skip reranking (no pre-computed tokens for a note → rank it at end). Anti-sycophancy fails → skip (log warning, return results without anti-sycophancy). Both search backends fail → empty list.

---

## Intelligence Layer

### link_generator.py — A-MEM Link Generation

Generates typed bidirectional links between notes using embedding cosine similarity.

- `LinkGenerator.__init__(db, zvec, embedder)` — stores references.
- `LinkGenerator.generate_links(note)` — for a given note, find top `LINK_CANDIDATE_LIMIT` nearest neighbors in Zvec, filter by cosine ≥ `LINK_SIMILARITY_THRESHOLD` (0.75), cap at `LINK_MAX_PER_NOTE` (5), create `"semantic"` links in SQLite.

### colbert_reranker.py — ColBERT Reranking

- `ColBERTReranker.__init__()` — loads `answerai-colbert-small-v1` model.
- `ColBERTReranker.encode_document(text) → bytes` — tokenize + forward pass → (num_tokens, 96) array → `.tobytes()`.
- `ColBERTReranker.rerank(query, candidates, top_n)` — encode query, load pre-computed doc tokens from SQLite, MaxSim scoring, return top_n.
- `ColBERTReranker.is_loaded() → bool` — model availability check.

### profile_generator.py — Static Peer Card

- `ProfileGenerator.__init__(db, deriver)` — uses Deriver API (DeepSeek V3.2) for generation.
- `ProfileGenerator.generate(peer_id)` → JSON dict with 4 sections, each a list of fact strings.
- Input: all notes with `durability="permanent"` for the peer.
- Output: stored in `peers.static_profile` and `peers.profile_updated_at`.
- **No expertise/skill level in static profile** — this causes models to over-anchor. Expertise lives in dynamic memory.
- User-set facts (confidence 1.0) override LLM observations (confidence 0.8).

---

## Background Processes

### Batch Dedup (dedup.py)

Runs once at the start of every Dreamer cycle. Pure math, no LLM calls. ~100ms for 100 buffered notes.

```
Buffered notes (is_buffered=1)
  ↓
Embed all (batch, via embedder.embed_documents)
  ↓
Pairwise cosine similarity
  ↓
Cluster notes where cosine > 0.85 (union-find, NOT NetworkX)
  ↓
For each cluster:
  - Pick highest-importance note as canonical
  - Sum evidence_count across cluster
  - Count unique_sessions_mentioned (one per session max)
  - Set canonical_note_id on merged notes
  - Set is_buffered=0 on all cluster members
  ↓
Compute importance for each canonical note:
  importance = emotional_weight × 0.6 + frequency_score × 0.4
  frequency_score = 1 - e^(-0.15 × unique_sessions_mentioned)
```

**`DedupProcessor.__init__(db, embedder)`**

**`DedupProcessor.run(peer_id) → DedupResult`**
- Fetch buffered notes → embed → cluster → merge → compute importance → return stats.
- `DedupResult` is a dataclass: `notes_processed: int`, `clusters_found: int`, `notes_merged: int`.
- If 0 or 1 buffered notes → return immediately (nothing to dedup).

**Union-find implementation**: simple in-module `_UnionFind` class with `find()` and `union()`. Avoids NetworkX dependency for this fast path.

### Gemini Client (gemini_client.py)

Thin wrapper around `google-genai` SDK for Batch API operations. The SDK is synchronous — wrap all calls with `asyncio.to_thread()`.

```python
from google import genai
from google.genai import types

client = genai.Client()  # Uses GEMINI_API_KEY env var
```

**`GeminiClient.__init__(api_key=None)`** — creates `genai.Client`. If api_key is None, reads from `GEMINI_API_KEY` env var.

**`GeminiClient.submit_batch(requests, display_name) → str`** — submits inline batch requests, returns job name. Uses `client.batches.create(model=GEMINI_MODEL, src=requests, config={"display_name": display_name})`.

**`GeminiClient.poll_until_done(job_name, poll_interval, max_time) → BatchJob`** — polls `client.batches.get()` until terminal state. Returns the job object.

**`GeminiClient.get_results(job) → list[dict]`** — extracts results from completed job. Handles both inline responses and file-based results.

**Job states**: `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`, `JOB_STATE_CANCELLED`, `JOB_STATE_EXPIRED`.

### Dreamer Prompts (prompts.py)

System prompts for each Dreamer task, as string constants. All require JSON output format.

**`LINK_GENERATION_PROMPT`** — given a set of notes, identify which pairs should be linked, with link type and strength. Return `{"links": [{"source_id": ..., "target_id": ..., "link_type": ..., "strength": ...}]}`.

**`PATTERN_DETECTION_PROMPT`** — given notes across sessions, identify cross-session trends, behavioral patterns. Return `{"patterns": [{"content": ..., "keywords": [...], "supporting_note_ids": [...]}]}`. Output notes are tagged `note_type: "inference"`, `provenance: "inferred"`.

**`CONTRADICTION_DETECTION_PROMPT`** — given pairs of notes with cosine > 0.80, determine if they contradict. Return `{"contradictions": [{"note_id_a": ..., "note_id_b": ..., "description": ...}]}`. Creates `"contradicts"` links.

**`PROFILE_UPDATE_PROMPT`** — given permanent notes and current profile, regenerate the static profile. Return `{"profile": {"identity": [...], "professional": [...], "communication_style": [...], "relationships": [...]}}`.

### Task Builder (task_builder.py)

Constructs Gemini Batch API request payloads from post-dedup notes.

**`build_link_requests(notes, existing_links) → list[dict]`** — creates batch requests for link generation. Groups notes into batches of ~20.

**`build_pattern_requests(notes, sessions) → list[dict]`** — creates batch requests for pattern detection across sessions.

**`build_contradiction_requests(candidate_pairs) → list[dict]`** — takes pairs of notes with cosine > 0.80, creates batch requests.

**`build_profile_request(permanent_notes, current_profile) → dict`** — single request for profile regeneration.

Each function returns request dicts in the format expected by `GeminiClient.submit_batch()`:
```python
{
    "contents": [{"parts": [{"text": prompt}], "role": "user"}],
    "system_instruction": {"parts": [{"text": system_prompt}]},
    "generation_config": {"temperature": temp, "response_mime_type": "application/json"},
}
```

### Processor (processor.py)

Processes Gemini batch results back into SQLite/Zvec.

**`DreamerProcessor.__init__(db, embedder, zvec)`**

**`DreamerProcessor.process_links(results) → int`** — parses link JSON, creates links in SQLite. Returns count created. Skips duplicates (UNIQUE constraint).

**`DreamerProcessor.process_patterns(results, peer_id) → int`** — creates inference notes from detected patterns. Tags with `note_type="inference"`, `provenance="inferred"`, `durability="contextual"`. Embeds and stores in Zvec. Returns count created.

**`DreamerProcessor.process_contradictions(results) → int`** — creates `"contradicts"` links between conflicting notes. Returns count created.

**`DreamerProcessor.process_profile(result, peer_id)`** — updates `peers.static_profile` and `peers.profile_updated_at`.

All processors are best-effort: failures are logged, never block the cycle.

### Orchestrator (orchestrator.py)

Coordinates a full Dreamer cycle.

**`DreamerOrchestrator.__init__(db, embedder, zvec, gemini_client, deriver)`**

**`DreamerOrchestrator.run_cycle(peer_id) → CycleResult`**

Steps:
1. **Dedup**: `DedupProcessor.run(peer_id)`. If 0 notes after dedup → return early.
2. **Find contradiction candidates**: pairwise cosine among buffered notes where similarity > 0.80.
3. **Build batch requests**: link generation, pattern detection, contradiction detection, profile update.
4. **Submit batch**: `gemini_client.submit_batch()`.
5. **Poll**: `gemini_client.poll_until_done()`.
6. **Process results**: links → patterns → contradictions → profile.
7. **MAGMA update**: update entity graph from new/modified notes.
8. **Return** `CycleResult` with counts.

`CycleResult` dataclass: `notes_deduped`, `links_created`, `patterns_found`, `contradictions_found`, `profile_updated: bool`.

If the Gemini batch job fails or times out, log the error and return a partial result (dedup still completes).

---

## MAGMA Multi-Graph (graph/magma.py)

Three orthogonal graphs stored in NetworkX, persisted to SQLite via `entity_mentions` table and `links` table.

**Entity graph**: nodes = entities (people, places, organizations, concepts), edges = co-occurrence in notes. Weight = number of shared notes.

**Temporal graph**: stored via `links` with `link_type="temporal"`. Nodes = notes, edges = temporal ordering within sessions.

**Causal graph**: stored via `links` with `link_type="causal"`. Created by the Dreamer.

### magma.py Methods

**`MAGMAGraph.__init__(db)`** — initializes empty NetworkX graphs.

**`MAGMAGraph.load(peer_id)`** — loads entity mentions from SQLite, builds in-memory NetworkX graph.

**`MAGMAGraph.extract_entities(text) → list[tuple[str, str]]`** — rule-based entity extraction (no LLM). Returns list of (entity_name, entity_type) tuples. Uses simple heuristics: capitalized multi-word sequences → person/organization, @-patterns → person, known location patterns → place. Best-effort; failure returns empty list.

**`MAGMAGraph.add_note_entities(note, entities)`** — adds entity mentions to SQLite and updates in-memory graph.

**`MAGMAGraph.get_related_entities(entity_name, peer_id, top_k=10) → list[str]`** — returns entities most connected to the given entity via graph centrality.

**`MAGMAGraph.get_entity_subgraph(entity_name, depth=2) → dict`** — BFS from entity node, returns subgraph as adjacency dict.

**`MAGMAGraph.get_communities(peer_id) → list[list[str]]`** — community detection (Louvain or label propagation) on the entity graph. Returns groups of related entities.

Entity extraction is called from `handle_derive()` after note creation. It runs synchronously and is best-effort — extraction failure never blocks note creation.

---

## Refinement Layer

### MemRL Q-Value Tracking (refinement/qvalue.py)

Learned utility score per memory. Updated by user feedback via temporal-difference updates. Integrated into the composite score as an additive factor after RRF fusion.

**`QValueTracker.__init__(db)`** — stores SQLiteStore reference.

**`QValueTracker.update(note_id, reward) → float`** — applies temporal-difference update:
```
Q_new = (1 - α) × Q_old + α × reward
```
Where `α = QVALUE_LEARNING_RATE` (0.1). Returns the new Q-value. Writes to SQLite.

**`QValueTracker.batch_update(note_ids, reward) → dict[str, float]`** — applies the same update to multiple notes. Returns `{note_id: new_q_value}`. Single transaction.

**`QValueTracker.normalize_qvalues(peer_id) → dict[str, float]`** — z-score normalization across the peer's recent notes. Uses `QVALUE_ZSCORE_WINDOW` (last 100 notes by created_at). If fewer than `QVALUE_ZSCORE_MIN_SAMPLES` (10), returns raw Q-values unchanged. Formula: `q_normalized = (q - mean) / max(std, 1e-8)`. Returns `{note_id: normalized_q}`.

**`QValueTracker.get_qvalue_boost(note_id, peer_id) → float`** — returns the Q-value contribution for composite scoring. Normalizes via z-score, then scales: `boost = normalized_q × QVALUE_WEIGHT_IN_COMPOSITE`. Clamped to [-0.3, 0.3] to prevent Q-values from dominating.

**Integration into retriever.py**: After computing composite_score (decay × provenance × fatigue × inference_discount), add Q-value boost:
```python
composite = rrf × decay × provenance × fatigue × inference_discount × agent_topic_penalty
qboost = qvalue_tracker.get_qvalue_boost(note.id, peer_id)
final_score = composite + qboost  # additive, not multiplicative
```

### User Feedback Ingestion (refinement/feedback.py)

Processes explicit and implicit feedback signals, routing them to Q-value updates and memory corrections.

**`FeedbackProcessor.__init__(db, qvalue_tracker, embedder, zvec)`** — stores references.

**`FeedbackProcessor.record_thumbs_up(peer_id, session_id, note_ids)`** — records feedback event, calls `qvalue_tracker.batch_update(note_ids, QVALUE_POSITIVE_REWARD)`.

**`FeedbackProcessor.record_thumbs_down(peer_id, session_id, note_ids)`** — records feedback event, calls `qvalue_tracker.batch_update(note_ids, QVALUE_NEGATIVE_REWARD)`.

**`FeedbackProcessor.record_implicit(peer_id, session_id, note_ids, signal_type)`** — `signal_type` is `"positive"` (engaged response) or `"negative"` (topic change). Uses `QVALUE_IMPLICIT_POSITIVE` or `QVALUE_IMPLICIT_NEGATIVE`.

**`FeedbackProcessor.record_correction(peer_id, session_id, note_id, correction_text) → Note`** — the memory correction flow:
1. Set `is_invalidated=1` on the contradicted note, drop its confidence to `FEEDBACK_CORRECTION_CONFIDENCE` (0.3).
2. Create a replacement note with `provenance="user_confirmed"`, `confidence=1.0`, `correction_of=note_id`.
3. Embed the replacement and store in Zvec.
4. Apply `QVALUE_NEGATIVE_REWARD` to the invalidated note.
5. Return the new replacement note.

**`FeedbackProcessor.get_feedback_history(peer_id, limit=50) → list[FeedbackEvent]`** — returns recent feedback for analytics.

### Memory Correction Flow (refinement/correction.py)

Handles conversational corrections where the user contradicts an existing memory.

**`CorrectionHandler.__init__(db, embedder, zvec, feedback_processor)`**

**`CorrectionHandler.detect_correction(user_message, surfaced_note_ids, peer_id) → CorrectionCandidate | None`** — uses embedding similarity between the user's message and surfaced notes. If any surfaced note has cosine similarity > 0.6 to the user message AND the user message contains correction signals (negation words: "no", "actually", "that's wrong", "that's not right", "I meant", "correction", "wrong", "incorrect"), returns a `CorrectionCandidate(note_id, correction_text, confidence)`. Pure heuristic — no LLM call.

**`CorrectionCandidate`** — dataclass: `note_id: str`, `correction_text: str`, `confidence: float`.

**`CorrectionHandler.apply_correction(candidate, peer_id, session_id) → Note`** — delegates to `feedback_processor.record_correction()`.

### Adaptive Retrieval Router (refinement/router.py)

Classifies queries as simple or complex to decide whether to use the standard retrieval pipeline or trigger the MemR3 reflective loop.

**`RetrievalRouter.__init__(embedder)`** — stores embedder reference.

**`RetrievalRouter.classify(query, candidates) → str`** — returns `"simple"` or `"complex"`. Classification signals:

1. **Score spread**: compute `max_score - min_score` across top candidates. Below `ROUTER_SIMPLE_THRESHOLD` (0.6) → likely complex (all results are mediocre matches).
2. **Embedding distance distribution**: compute the mean and std of cosine distances between query and top results. If mean distance > `ROUTER_DISTANCE_PERCENTILE`-th percentile of the peer's historical distances → complex (query is unlike stored memories).
3. **Query structure heuristics**: multi-hop indicators ("how many", "compare", "between X and Y", temporal references spanning multiple events) → complex. Direct lookup patterns ("what is", "who is", single-entity questions) → simple.
4. **Candidate count**: fewer than `ROUTER_MIN_CANDIDATES` (3) results above a minimum score → complex (insufficient evidence).

Returns `"complex"` if 2+ signals fire, `"simple"` otherwise. Default to `"simple"` on any classification error.

**Expected ratio**: ~88% simple, ~12% complex based on typical conversational query distributions.

### MemR3 Reflective Loop (refinement/reflective.py)

Closed-loop retrieval controller for complex queries. Wraps the existing Retriever and adds an evidence-gap tracker with retrieve/reflect/answer routing.

**`ReflectiveRetriever.__init__(retriever, router, deriver_client, embedder)`** — stores references. `deriver_client` is used for the reflect step (DeepSeek V3.2 via NousResearch, same endpoint as the Deriver).

**`ReflectiveRetriever.retrieve(query, peer_id, limit=10) → ReflectiveResult`** — the main entry point:

1. Run initial retrieval via `self._retriever.retrieve(query, peer_id)`.
2. Call `router.classify(query, results)`. If `"simple"` → return results directly as `ReflectiveResult`.
3. If `"complex"` → enter the reflective loop.

**Reflective loop** (max `MEMR3_MAX_ITERATIONS` = 3 iterations):

```
Initialize:
  evidence = {}     # note_id → content (what we know)
  gaps = [query]    # what we still need (starts as the original query)
  mask_ids = set()  # IDs already retrieved (to prevent re-retrieval)
  iteration = 0

Loop:
  1. RETRIEVE: run retriever with refined query (gaps → query reformulation)
     - Mask: exclude results where cosine(result, any masked) > MEMR3_MASK_THRESHOLD
     - Add non-masked results to evidence, add their IDs to mask_ids

  2. REFLECT: call DeepSeek V3.2 with prompt containing:
     - Original query
     - Current evidence (all gathered so far)
     - Current gaps
     - Ask: "What evidence is confirmed? What is still missing? Should we retrieve more or answer now?"
     - Output: updated evidence dict, updated gaps list, action ("retrieve" | "answer")
     - Temperature: MEMR3_REFLECT_TEMPERATURE (0.1)
     - Retry: MEMR3_REFLECT_MAX_RETRIES (2) with same backoff as Deriver

  3. ROUTE:
     - If action == "answer" OR gaps is empty OR iteration >= max → break
     - If action == "retrieve" → reformulate query from gaps, continue loop

  iteration += 1
```

**`ReflectiveResult`** — Pydantic model: `results: list[RetrievalResult]`, `is_complex: bool`, `iterations: int`, `evidence: dict[str, str]`, `gaps: list[str]`, `latency_ms: float`.

**Reflect prompt** (stored as a constant in `refinement/reflective.py`):
```
You are analyzing retrieved memories to answer a question.

Question: {query}

Evidence gathered so far:
{evidence_text}

Gaps (information still needed):
{gaps_text}

Respond with JSON only:
{
  "evidence": {"note_id": "confirmed fact", ...},
  "gaps": ["what is still missing", ...],
  "action": "retrieve" | "answer",
  "refined_query": "if action is retrieve, the next search query"
}
```

**Error handling**: If the reflect LLM call fails, break the loop and return whatever evidence was gathered. The reflective loop is best-effort — failure falls back to standard retrieval results.

### Anti-Sycophancy Guards (refinement/antisycophancy.py)

Five defensive layers preventing memory-powered personalization from creating self-reinforcing feedback loops.

**Layer 1 — Provenance-weighted scoring** (implemented in scorer.py): organic=1.0, user_confirmed=0.8, agent_prompted=0.5, inferred=0.3.

**Layer 2 — Topic cluster diversity cap**:

**`enforce_topic_diversity(results, embedder, cap=ANTISYC_TOPIC_CLUSTER_CAP) → list[RetrievalResult]`** — clusters the result set by embedding similarity (threshold `ANTISYC_CLUSTER_SIMILARITY` = 0.75, same union-find approach as dedup). If any cluster exceeds `cap` (40%) of the result set, drops the lowest-scored members of that cluster until the cap is met. Dropped slots are filled by the next-best results from other clusters if available. If only one cluster exists, return as-is — do not drop below 1 result.

**Layer 3 — Epsilon-greedy exploration**:

**`apply_epsilon_exploration(results, peer_id, db, epsilon=ANTISYC_EPSILON) → list[RetrievalResult]`** — with probability `epsilon` (0.1), replace the lowest-scored result in the list with a random note from the peer's memory (excluding notes already in the result set). The random note must have `is_invalidated=0`. If the random pick fails (no eligible notes), skip exploration for this query. Returns the modified results. Random notes are wrapped in RetrievalResult with `score=0.0` and `source="exploration"`.

**Layer 4 — Anti-sycophancy system prompt**:

**`generate_antisycophancy_prompt(profile_text, memory_count) → str`** — returns a system prompt fragment to append when memory context is injected:
```
The following memories are provided for context only. Do not:
- Assume these memories represent the user's current views or interests
- Proactively introduce topics merely because they appear in memory
- Agree with positions in memory without the user raising them
- Treat memory-derived facts as more reliable than the user's current statements
If the user contradicts a memory, trust the user's current statement.
```

**Layer 5 — Agent-topic decay flagging**:

**`AgentTopicTracker.__init__(db)`**

**`AgentTopicTracker.track_session(peer_id, session_id, organic_topics, agent_topics)`** — `organic_topics` = topic embeddings from user-originated notes this session. `agent_topics` = topic embeddings from agent-originated context. For each agent topic: if it matches an existing tracked topic (cosine > 0.75), increment `consecutive_agent_sessions`. If it also appears in `organic_topics`, reset `consecutive_agent_sessions` to 0 and update `last_organic_session_id`. If `consecutive_agent_sessions >= ANTISYC_AGENT_TOPIC_DECAY_SESSIONS` (3), set `flagged_for_decay=1`.

**`AgentTopicTracker.get_flagged_notes(peer_id) → list[str]`** — returns note IDs associated with flagged topics. These notes get an additional decay penalty applied during scoring (0.5× via `compute_agent_topic_penalty`).

---

## Models

### RetrievalResult (models.py)

```python
class RetrievalResult(BaseModel):
    note: Note
    score: float              # Final composite score
    rrf_score: float          # Raw RRF before multipliers
    decay_strength: float
    provenance_weight: float
    fatigue_factor: float
    inference_discount: float = 1.0
    colbert_score: float | None = None
    q_value_boost: float = 0.0
    agent_topic_penalty: float = 1.0
    source: str               # "fts" | "vector" | "both" | "link" | "exploration"
```

### ReflectiveResult (models.py)

```python
class ReflectiveResult(BaseModel):
    results: list[RetrievalResult]
    is_complex: bool = False
    iterations: int = 0
    evidence: dict[str, str] = {}
    gaps: list[str] = []
    latency_ms: float = 0.0
```

### FeedbackEvent (models.py)

```python
class FeedbackEvent(BaseModel):
    id: str
    peer_id: str
    session_id: str | None = None
    note_ids: list[str]
    feedback_type: str   # "thumbs_up" | "thumbs_down" | "correction" | "implicit_positive" | "implicit_negative"
    correction_text: str | None = None
    created_at: str
```

---

## Latency Targets

| Path | Target | Notes |
|------|--------|-------|
| Fast path (write + enqueue) | < 5ms | User never waits past this |
| Full async write pipeline | < 5s | Not user-facing |
| FTS5 search | < 1ms | SQLite in-process |
| Zvec vector search | ~2-5ms | HNSW in-process |
| Query embedding | ~10ms | Single call |
| RRF + scoring | < 1ms | Pure math, ~60 candidates |
| ColBERT rerank | ~5-7ms | Query encode ~5ms + MaxSim ~1-2ms (pre-computed docs) |
| MMR dedup | ~20-50ms | Batch embed + dot products |
| Anti-sycophancy (topic diversity) | < 5ms | Pairwise cosine on ~10 results |
| Anti-sycophancy (epsilon exploration) | < 2ms | Random pick from SQLite |
| Q-value update (single note) | < 1ms | SQLite write |
| Q-value normalization | < 5ms | Read + compute |
| Router classification | < 2ms | Pure math + heuristics |
| **Full retrieval (simple, with ColBERT)** | **< 100ms** | **User-facing, pre-computed tokens** |
| **Full retrieval (simple, without ColBERT)** | **< 80ms** | **Fallback** |
| MemR3 reflective loop (per iteration) | ~1-1.5s | One LLM call for reflect |
| **Full retrieval (complex, MemR3)** | **< 4s** | **1-3 iterations** |
| Feedback recording | < 2ms | SQLite insert + Q-value update |
| Memory correction flow | < 50ms | Invalidate + create + embed |
| Batch dedup (100 notes) | ~100ms | Start of Dreamer cycle |
| Dreamer cycle | < 24h | Batch API SLO, typically much faster |

---

## Environment Variables

```bash
export NOUSRESEARCH_API_KEY="your-key"   # Required for Deriver and MemR3 reflect step
export GEMINI_API_KEY="your-key"         # Required for Dreamer
```

---

## Tests

All use `tmp_path`. All API calls mocked.

### Foundation
- **test_sqlite_store.py**: schema, CRUD, FTS5 triggers, task queue atomic dequeue, dead-letter.
- **test_embedder.py**: 384-dim, deterministic, doc vs query prefix produces different vectors.
- **test_zvec_store.py**: insert+query, batch, delete, empty query.
- **test_memory_api.py**: full round trip peer→session→message→note→search (keyword, vector, hybrid).

### Write Pipeline
- **test_worker.py**: run_once, empty queue, exception handling, dead-letter.
- **test_intake.py**: user enqueues, assistant doesn't, preceding context, FTS5 findable.
- **test_deriver.py**: extraction, confirmation, empty, scorer fields, retry on 429, garbage JSON.
- **test_write_pipeline.py**: full round trip ingest→derive→notes in SQLite+Zvec+FTS5.

### Retrieval
- **test_scorer.py**: decay disabled < 100, ramps 100–1000, high-importance floor, provenance weights, fatigue decreasing, composite multiplies, Q-value boost additive, agent topic penalty halves score.
- **test_fusion.py**: RRF overlapping > non-overlapping, MMR drops > 0.90, order preserved, missing embeddings auto-accepted.
- **test_retriever.py**: basic retrieval, source="both", peer isolation, access tracking, empty results, fallbacks, MMR dedup, provenance/decay ordering, limit respected, invalidated notes excluded.
- **test_retrieval_pipeline.py**: end-to-end write-then-read, multiple sessions, surfacing fatigue, mixed provenance.

### Intelligence Layer
- **test_link_generator.py**: generates links above threshold, skips below, caps at max per note, bidirectional query works, no duplicate links.
- **test_colbert_reranker.py**: encode_document returns bytes of correct shape, rerank with pre-computed tokens returns reranked ids, graceful failure returns original order, notes without tokens ranked at end, top_n respected.
- **test_profile_generator.py**: generates valid 4-section profile, stores in peer, skips empty notes, no expertise in profile.

### Background Processes
- **test_dedup.py**: clusters similar notes, merges canonical, sums evidence_count, computes importance, handles single notes, handles no buffered notes, union-find correctness.
- **test_gemini_client.py**: submit_batch returns job name, poll_until_done handles states (success/fail/expired), get_results parses inline and file responses. All calls mocked.
- **test_dreamer_orchestrator.py**: full cycle mock (dedup → submit → poll → process), handles empty buffer, handles batch failure gracefully, partial results on timeout.
- **test_magma.py**: extract_entities finds capitalized names, add_note_entities persists to SQLite, get_related_entities returns connected nodes, get_communities returns groups, empty graph returns empty lists.
- **test_dreamer_pipeline.py**: end-to-end with mocked Gemini — buffer notes → dedup → dream → verify links/patterns/profile created. Entity mentions stored. Contradictions create links.

### Refinement
- **test_qvalue.py**: default Q-value is 0.0, positive update increases, negative decreases asymmetrically, batch update applies to all, z-score normalization with sufficient samples, normalization skipped below min_samples, Q-value boost clamped to [-0.3, 0.3], composite score changes with Q-value.
- **test_reflective.py**: simple query skips loop, complex query triggers loop, evidence-gap tracker accumulates across iterations, mask prevents re-retrieval of same notes, early stopping when gaps empty, early stopping at max iterations, reflect LLM failure falls back gracefully, latency tracked.
- **test_router.py**: high score spread → simple, low score spread → complex, multi-hop query patterns → complex, single-entity patterns → simple, few candidates → complex, classification error defaults to simple.
- **test_feedback.py**: thumbs up increases Q-value, thumbs down decreases Q-value (asymmetric), correction invalidates note and creates replacement, correction note has user_confirmed provenance, implicit signals apply small deltas, feedback events persisted to SQLite.
- **test_correction.py**: detect_correction finds contradiction with signal words, detect_correction returns None without signal words, detect_correction returns None with low similarity, apply_correction delegates correctly.
- **test_antisycophancy.py**: topic cluster cap enforced at 40%, epsilon exploration replaces lowest with random, agent-topic tracker increments consecutive sessions, agent-topic tracker resets on organic mention, flagged notes get penalty in scoring, epsilon exploration skips when no eligible random notes.
- **test_refinement_pipeline.py**: end-to-end feedback → Q-value → next retrieval reranks correctly, correction flow → old note deprioritized + new note surfaces, anti-sycophancy layers compose without breaking retrieval order.

---

## Design Principles

1. **User never waits for memory processing.** Write path is async; read path is < 100ms.
2. **Raw data is immutable.** Stream 1 observations are ground truth. Dreamer creates Stream 3 inferences, never modifies Stream 1.
3. **Frequency ≠ importance.** Two-dimensional scoring prevents topic domination.
4. **Decay scores, never deletes.** All data preserved; relevance is temporal.
5. **Agent responses are context, not memory.** Read but never stored as user attributes.
6. **Graceful degradation.** If one search backend, the Dreamer, or the reflective loop fails, the system continues.
7. **No LLM calls in the standard read path.** ColBERT is a local model, not an API call. MemR3 only fires for complex queries (~12%).
8. **No expertise in static profile.** Prevents model over-anchoring. Expertise lives in dynamic notes.
9. **Batch dedup before Dreamer.** Preserves frequency as an importance signal while preventing the Dreamer from being overwhelmed by repetition.
10. **Entity extraction is best-effort.** Rule-based, no LLM. Failure never blocks note creation.
11. **Q-values are additive, not multiplicative.** Prevents a single bad Q-value from zeroing out an otherwise relevant memory.
12. **The reflective loop is best-effort.** LLM failure in the reflect step falls back to standard retrieval, never blocks the response.
13. **Anti-sycophancy guards compose independently.** Each layer can be disabled without affecting the others.
14. **Memory corrections trust the user.** A user correction always wins over LLM-derived observations.
15. **Routing defaults to simple.** Classification errors never trigger the expensive reflective loop.
