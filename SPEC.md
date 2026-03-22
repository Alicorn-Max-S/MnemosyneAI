# Mnemosyne — Spec

AI agent memory system. SQLite + FTS5 + Zvec for storage/search. Async write pipeline (Deriver via DeepSeek V3.2). Retrieval pipeline (parallel search, RRF fusion, scoring, MMR dedup). Intelligence layer (A-MEM links, ColBERT reranking, static profile). Background processes (batch dedup, Dreamer via Gemini 3 Flash Batch API, MAGMA multi-graph).

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
│   ├── models.py               # Pydantic: Peer, Session, Message, Note, Link, TaskItem, RetrievalResult
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
│   └── utils/
│       └── ids.py              # ULID generation
└── tests/
    ├── conftest.py
    ├── test_sqlite_store.py
    ├── test_zvec_store.py
    ├── test_embedder.py
    ├── test_memory_api.py
    ├── test_worker.py
    ├── test_intake.py
    ├── test_deriver.py
    ├── test_write_pipeline.py
    ├── test_scorer.py
    ├── test_fusion.py
    ├── test_retriever.py
    ├── test_retrieval_pipeline.py
    ├── test_link_generator.py
    ├── test_colbert_reranker.py
    ├── test_profile_generator.py
    ├── test_dedup.py
    ├── test_gemini_client.py
    ├── test_dreamer_orchestrator.py
    ├── test_magma.py
    └── test_dreamer_pipeline.py  # End-to-end: buffer → dedup → dream → links + profile
```

---

## Dependencies

```toml
[project]
name = "mnemosyne"
version = "0.5.0"
requires-python = ">=3.11"
dependencies = [
    "aiosqlite>=0.20.0",
    "zvec>=0.2.0",
    "sentence-transformers[onnx]>=5.0.0",
    "onnxruntime>=1.18.0",
    "einops>=0.8.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
    "python-ulid>=3.0.0",
    "numpy>=1.26.0",
    "pylate>=1.3.0",
    "google-genai>=1.0.0",
    "networkx>=3.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0"]
```

Do NOT add fastapi, uvicorn, ragatouille, or rerankers. Those are not needed.

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

**Tables:** config, peers, sessions, messages, notes, links, task_queue, entity_mentions. See `sqlite_store.py` for full DDL. The critical table is **notes**:

| Column group | Columns |
|---|---|
| Identity | id, peer_id, session_id, source_message_id |
| Content | content, context_description, keywords (JSON), tags (JSON) |
| Classification | note_type, provenance, durability |
| Scoring | emotional_weight, importance, confidence |
| Frequency | evidence_count, unique_sessions_mentioned |
| Retrieval state | q_value, access_count, last_accessed_at, times_surfaced, decay_score |
| Pipeline state | is_buffered, canonical_note_id, zvec_id |
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

---

## Embedder / ZvecStore / MemoryAPI

Already implemented. See source files for details.

**Key facts:**
- `Embedder.embed_query(text)` prepends `"search_query: "`, returns 384-dim L2-normalized list[float].
- `Embedder.embed_documents(texts)` batch version for MMR embedding.
- `ZvecStore.search(query_embedding, top_k)` returns `[{"id": str, "score": float}, ...]`.
- Zvec has **no scalar fields/filtering** — returns results across all peers. Post-filter by peer_id after hydrating from SQLite.
- MemoryAPI gains `retrieve(query, peer_id, limit)` which delegates to the Retriever, and `run_dreamer_cycle(peer_id)` which delegates to the Dreamer orchestrator.

---

## Write Pipeline

1. **intake.py**: `ingest_message()` writes to SQLite + enqueues "derive" task for user messages. Assistant messages stored but not derived. < 5ms.
2. **deriver.py**: `Deriver.extract()` → atomic facts from user message. `Deriver.score()` → tags each note with emotional_weight, provenance, durability, keywords, tags, context_description. Both call DeepSeek V3.2 via NousResearch (httpx, not openai SDK). Retry 3x with 1/2/4s backoff.
3. **handlers.py**: `handle_derive()` wires extract → score → embed → store (SQLite + Zvec). Also pre-computes ColBERT token embeddings and stores them, and extracts entities for MAGMA.
4. **worker.py**: polls task queue, dispatches to handlers, manages retries/dead-letter.

**Critical rules**: Agent responses are read for context but NEVER extracted from. User confirmations get `provenance: "user_confirmed"`. Notes stored with `is_buffered=1` for the Dreamer.

---

## Retrieval Pipeline

No LLM calls. Pure local computation. Target: < 100ms with ColBERT, < 80ms without.

```
Query
  ├→ embed_query() ──→ Zvec search (top 30)
  ├→ FTS5 search (top 30)
  ├→ Link expansion on top vector hits (BFS depth 1)
  │         ↓
  │    Hydrate from SQLite + peer_id filter
  │         ↓
  │    RRF Fusion (k=60) across all three streams
  │         ↓
  │    Score multipliers: decay × provenance × fatigue × inference_discount
  │         ↓
  │    ColBERT rerank top 50 → top 10
  │    (encode query tokens, load pre-computed doc tokens from SQLite, MaxSim)
  │         ↓
  │    MMR Dedup (cosine > 0.90 → drop lower-scored)
  │         ↓
  │    Top-N (default 10) → record_access() → return
```

### scorer.py — Pure scoring functions

All functions are stateless, no I/O.

- `compute_decay_strength(importance, days_since_access, access_count, total_memories)` — Ebbinghaus decay with importance-weighted lambda, access boost, ramp from 100–1000 memories, floor of 0.3 for high importance.
- `compute_provenance_weight(provenance)` — Lookup from `PROVENANCE_WEIGHTS`.
- `compute_surfacing_fatigue(times_surfaced)` — `1 / (1 + 0.1 * times_surfaced)`
- `compute_inference_discount(note_type)` — "inference" → 0.7, else → 1.0.
- `compute_composite_score(rrf_score, decay, provenance, fatigue, inference_discount)` — multiply all factors.

### fusion.py — RRF + MMR

- `rrf_fuse(ranked_lists, k=60)` — 0-indexed ranks. Notes in multiple lists get multiple contributions.
- `mmr_dedup(scored_ids, embeddings, threshold=0.90)` — cosine = dot product for L2-normalized vectors. Missing embeddings → auto-accept.

### retriever.py — Orchestrator

Steps: parallel search → hydrate → RRF → score → ColBERT rerank (query encode + pre-computed MaxSim) → MMR dedup → truncate → record access → return.

**Error handling**: Zvec fails → FTS-only. FTS fails → vector-only. ColBERT fails → skip reranking (no pre-computed tokens for a note → rank it at end). Both search backends fail → empty list.

---

## Intelligence Layer

### link_generator.py — A-MEM Link Generation

Generates typed bidirectional links between notes using embedding cosine similarity.

- `LinkGenerator.__init__(db, zvec, embedder)` — stores references.
- `LinkGenerator.generate_links(note)` — for a given note, find top `LINK_CANDIDATE_LIMIT` nearest neighbors in Zvec, filter by cosine ≥ `LINK_SIMILARITY_THRESHOLD` (0.75), cap at `LINK_MAX_PER_NOTE` (5), create `"semantic"` links in SQLite. No LLM call — pure embedding math.
- Called from `handle_derive()` after note creation. Failure does not block note persistence.

### colbert_reranker.py — ColBERT Reranking (Pre-computed Token Embeddings)

Uses `pylate` library with `answerdotai/answerai-colbert-small-v1` (33M params, 96-dim tokens, ~130MB).

**Write-time (pre-computation):**
- `ColBERTReranker.__init__()` — loads model via `pylate.models.ColBERT(COLBERT_MODEL)`.
- `ColBERTReranker.encode_document(text) → bytes` — encodes a note's text into token-level embeddings using `model.encode([text], is_query=False)`. Returns the numpy array serialized to bytes via `ndarray.tobytes()` for SQLite BLOB storage.
- `ColBERTReranker.encode_documents(texts) → list[bytes]` — batch version.
- Called from `handle_derive()` after note creation. Token embeddings stored in `colbert_tokens` table. Failure does not block note persistence.

**Read-time (reranking):**
- `ColBERTReranker.rerank(query, candidate_note_ids, db, top_n=10) → list[str]` — encodes query tokens via `model.encode([query], is_query=True)`, loads pre-computed document token BLOBs from `db.get_colbert_tokens(candidate_note_ids)`, reconstructs numpy arrays, computes MaxSim scores via `pylate.rank.rerank()`, returns reranked note_ids.
- Only the query requires a forward pass (~5ms). Document scoring is pure MaxSim matrix ops (~1-2ms for 50 candidates).
- Notes without pre-computed tokens (e.g., legacy notes from before Phase 5) are auto-ranked at the end of the list.

**Storage:** each note's token embeddings are ~(num_tokens × 96 × 4) bytes. A typical 50-token note ≈ 19KB raw. With numpy float32.

- Model loaded lazily on first use. Failure → return original ranking (graceful degradation).
- pylate is sync — wrap with `asyncio.to_thread()`.

### profile_generator.py — Static Peer Card

Generates a ~30-fact static profile organized into 4 sections: identity, professional, communication_style, relationships.

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
    source: str               # "fts" | "vector" | "both" | "link"
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
| **Full retrieval (with ColBERT)** | **< 100ms** | **User-facing, pre-computed tokens** |
| **Full retrieval (without ColBERT)** | **< 80ms** | **Fallback** |
| Batch dedup (100 notes) | ~100ms | Start of Dreamer cycle |
| Dreamer cycle | < 24h | Batch API SLO, typically much faster |

---

## Environment Variables

```bash
export NOUSRESEARCH_API_KEY="your-key"   # Required for Deriver
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
- **test_scorer.py**: decay disabled < 100, ramps 100–1000, high-importance floor, provenance weights, fatigue decreasing, composite multiplies.
- **test_fusion.py**: RRF overlapping > non-overlapping, MMR drops > 0.90, order preserved, missing embeddings auto-accepted.
- **test_retriever.py**: basic retrieval, source="both", peer isolation, access tracking, empty results, fallbacks, MMR dedup, provenance/decay ordering, limit respected.
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

---

## Design Principles

1. **User never waits for memory processing.** Write path is async; read path is < 100ms.
2. **Raw data is immutable.** Stream 1 observations are ground truth. Dreamer creates Stream 3 inferences, never modifies Stream 1.
3. **Frequency ≠ importance.** Two-dimensional scoring prevents topic domination.
4. **Decay scores, never deletes.** All data preserved; relevance is temporal.
5. **Agent responses are context, not memory.** Read but never stored as user attributes.
6. **Graceful degradation.** If one search backend or the Dreamer fails, the system continues.
7. **No LLM calls in the read path.** ColBERT is a local model, not an API call.
8. **No expertise in static profile.** Prevents model over-anchoring. Expertise lives in dynamic notes.
9. **Batch dedup before Dreamer.** Preserves frequency as an importance signal while preventing the Dreamer from being overwhelmed by repetition.
10. **Entity extraction is best-effort.** Rule-based, no LLM. Failure never blocks note creation.
