"""Zvec vector store for HNSW-based similarity search."""

import logging
import os

import zvec

from mnemosyne.config import EMBEDDING_DIM, ZVEC_COLLECTION_DIR, ZVEC_COLLECTION_NAME

logger = logging.getLogger(__name__)


class ZvecStore:
    """Manages a Zvec collection for storing and querying note embeddings."""

    def __init__(self, data_dir: str) -> None:
        """Open or create the Zvec collection at data_dir."""
        self._path = os.path.join(data_dir, ZVEC_COLLECTION_DIR)
        schema = zvec.CollectionSchema(
            name=ZVEC_COLLECTION_NAME,
            vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, EMBEDDING_DIM),
        )
        try:
            self._collection = zvec.open(path=self._path)
            logger.info("Opened existing Zvec collection at %s", self._path)
        except Exception:
            self._collection = zvec.create_and_open(path=self._path, schema=schema)
            logger.info("Created new Zvec collection at %s", self._path)

    def insert(self, note_id: str, embedding: list[float]) -> None:
        """Insert a single embedding and rebuild the HNSW index."""
        try:
            self._collection.insert([zvec.Doc(id=note_id, vectors={"embedding": embedding})])
            self._collection.optimize()
            logger.debug("Inserted and optimized note %s", note_id)
        except Exception:
            logger.exception("Failed to insert note %s", note_id)
            raise

    def insert_batch(self, items: list[tuple[str, list[float]]]) -> None:
        """Insert multiple embeddings and rebuild the HNSW index."""
        try:
            docs = [zvec.Doc(id=nid, vectors={"embedding": emb}) for nid, emb in items]
            self._collection.insert(docs)
            self._collection.optimize()
            logger.debug("Batch inserted %d notes", len(items))
        except Exception:
            logger.exception("Failed batch insert of %d notes", len(items))
            raise

    def search(self, query_embedding: list[float], top_k: int = 20) -> list[dict]:
        """Search for nearest neighbors, returning list of {id, score} dicts."""
        try:
            results = self._collection.query(
                zvec.VectorQuery("embedding", vector=query_embedding),
                topk=top_k,
            )
            return [{"id": r.id, "score": r.score} for r in results]
        except Exception:
            logger.exception("Vector search failed")
            return []

    def delete(self, note_id: str) -> None:
        """Delete a document from the collection."""
        try:
            self._collection.delete(ids=note_id)
            logger.debug("Deleted note %s from vector store", note_id)
        except Exception:
            logger.exception("Failed to delete note %s", note_id)
            raise

    def optimize(self) -> None:
        """Rebuild the HNSW index."""
        self._collection.optimize()

    def stats(self) -> dict:
        """Return collection info."""
        try:
            info = self._collection.info()
            return {"path": self._path, "info": str(info)}
        except Exception:
            logger.exception("Failed to get collection stats")
            return {"path": self._path, "info": "unavailable"}
