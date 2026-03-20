# Phase 4 Complete — Intelligence Layer

## New Files

### `mnemosyne/intelligence/`
- `__init__.py` — public exports: `ColBERTReranker`, `Linker`, `Profiler`
- `reranker.py` — ColBERT-based reranker using `answerdotai/answerai-colbert-small-v1`
- `linker.py` — A-MEM inspired automatic linker (semantic similarity links)
- `profiler.py` — Static peer profiler aggregating memory into `PeerProfile`

### `tests/`
- `test_reranker.py` — ColBERT reranker unit tests
- `test_linker.py` — Linker unit tests
- `test_profiler.py` — Profiler unit tests
- `test_link_expansion.py` — Link expansion in retrieval pipeline
- `test_intelligence_pipeline.py` — End-to-end intelligence pipeline integration tests

## Modified Files

- `mnemosyne/config.py` — Phase 4 constants (ColBERT model name, rerank top-k, link thresholds, profiler settings)
- `mnemosyne/models.py` — Added `PeerProfile` model, `colbert_score` field on scored results
- `mnemosyne/db/sqlite_store.py` — `links` and `peer_profiles` tables, new CRUD methods for links and profiles
- `mnemosyne/pipeline/__init__.py` — Linker integration into write pipeline
- `mnemosyne/pipeline/handlers.py` — Linker handler in pipeline processing
- `mnemosyne/retrieval/__init__.py` — Link expansion + ColBERT reranking exports
- `mnemosyne/retrieval/retriever.py` — Link expansion and ColBERT reranking in retrieval flow
- `mnemosyne/vectors/embedder.py` — Batch embedding support
- `pyproject.toml` — Version bump to 0.4.0, added `numpy>=1.26.0` dependency
- `tests/test_write_pipeline.py` — Updated for linker support in pipeline

## Running Phase 4 Tests

```bash
python3 -m pytest tests/test_reranker.py tests/test_linker.py tests/test_profiler.py tests/test_link_expansion.py tests/test_intelligence_pipeline.py -v
```

Full suite (197 tests):

```bash
python3 -m pytest tests/ -v
```

## Known Limitations

- ColBERT model loads ~2-3s on first use (~130MB download)
- Linker only creates "semantic" links; causal/temporal link types are Phase 5
- Profiler requires LLM API access for summary generation
- No GPU acceleration — CPU-only inference on target VPS

## Usage Example

```python
from mnemosyne.intelligence import ColBERTReranker
from mnemosyne.retrieval import Retriever

# Retriever with ColBERT reranking
retriever = Retriever(sqlite_store, zvec_store, embedder, reranker=ColBERTReranker())
results = await retriever.search("what did Alice say about the project?", peer_id="peer_01")
# Results are reranked by ColBERT for higher relevance
```
