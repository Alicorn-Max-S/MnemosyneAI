#!/usr/bin/env python3
"""Live end-to-end pipeline test using real DeepSeek v3.2 via NousResearch API.

Exercises: Embedder (ONNX) -> add_note -> Deriver (extract + score) -> Retriever.

Usage:
    python3 scripts/test_live_pipeline.py

Requires NOUSRESEARCH_API_KEY in .env or environment.
"""

import asyncio
import os
import sys
import tempfile

# Load .env if present
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

from mnemosyne.api.memory_api import MemoryAPI
from mnemosyne.pipeline.deriver import Deriver
from mnemosyne.vectors.embedder import Embedder


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def status(label: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    msg = f"  [{tag}] {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if not ok:
        sys.exit(1)


async def main() -> None:
    api_key = os.environ.get("NOUSRESEARCH_API_KEY", "")
    if not api_key:
        print("ERROR: NOUSRESEARCH_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        # ── 1. Embedder ──────────────────────────────────────────
        print("\n== Embedder ==")
        embedder = Embedder()
        status("ONNX backend loaded", embedder.backend == "onnx", embedder.backend)
        status("Dimension is 384", embedder.dimension == 384, str(embedder.dimension))

        # ── 2. MemoryAPI: add notes ──────────────────────────────
        print("\n== MemoryAPI ==")
        api = MemoryAPI(data_dir=tmp, embedder=embedder)
        await api.initialize()
        status("MemoryAPI initialized", True)

        peer = await api.create_peer("live_test_user")
        status("Peer created", peer is not None, peer.id)

        note1 = await api.add_note(peer.id, "My golden retriever Buddy loves swimming in the lake")
        note2 = await api.add_note(peer.id, "I have a severe allergy to shellfish")
        note3 = await api.add_note(
            peer.id,
            "User probably prefers outdoor activities based on context",
            note_type="inference",
        )
        status("3 notes added (2 organic + 1 inference)", True)

        # ── 3. Deriver: extract + score ──────────────────────────
        print("\n== Deriver (DeepSeek v3.2) ==")
        deriver = Deriver(api_key=api_key)
        status("Deriver created", True, f"model={deriver._model}")

        test_message = "I just got back from a camping trip in Yosemite with my family. We saw bears!"
        preceding = [
            {"role": "assistant", "content": "How was your weekend?"},
        ]

        notes = await deriver.extract(test_message, preceding)
        status(
            "Extract returned notes",
            len(notes) > 0,
            f"{len(notes)} facts extracted",
        )
        for i, n in enumerate(notes):
            print(f"    [{i+1}] {n.get('text', '???')}")

        scored = await deriver.score(notes)
        status(
            "Score returned metadata",
            len(scored) > 0 and "provenance" in scored[0],
            f"{len(scored)} scored",
        )
        for i, s in enumerate(scored):
            print(
                f"    [{i+1}] {s.get('text', '???')}"
                f" | prov={s.get('provenance')} dur={s.get('durability')}"
                f" | kw={s.get('keywords')}"
            )

        # Store the derived notes
        for s in scored:
            await api.add_note(
                peer.id,
                s["text"],
                provenance=s.get("provenance", "organic"),
                durability=s.get("durability", "contextual"),
                emotional_weight=s.get("emotional_weight", 0.5),
                keywords=s.get("keywords"),
                tags=s.get("tags"),
                context_description=s.get("context_description"),
            )
        status("Derived notes stored", True, f"{len(scored)} added")

        # ── 4. Retriever ─────────────────────────────────────────
        print("\n== Retriever ==")

        results = await api.retrieve("dog swimming", peer.id)
        status("Retrieve 'dog swimming'", len(results) > 0, f"{len(results)} results")
        if results:
            r = results[0]
            print(f"    Top: \"{r.note.content[:80]}\"")
            print(f"    score={r.score:.4f} rrf={r.rrf_score:.4f} source={r.source}")
            print(f"    decay={r.decay_strength:.4f} prov={r.provenance_weight:.2f}"
                  f" fatigue={r.fatigue_factor:.4f} inf_disc={r.inference_discount:.2f}")
            status(
                "Top result mentions dog/buddy/swim",
                any(w in r.note.content.lower() for w in ("dog", "buddy", "swim")),
            )

        results2 = await api.retrieve("camping Yosemite bears", peer.id)
        status(
            "Retrieve 'camping Yosemite'",
            len(results2) > 0,
            f"{len(results2)} results",
        )
        if results2:
            print(f"    Top: \"{results2[0].note.content[:80]}\"")

        results3 = await api.retrieve("allergy food", peer.id)
        status(
            "Retrieve 'allergy food'",
            len(results3) > 0,
            f"{len(results3)} results",
        )
        if results3:
            print(f"    Top: \"{results3[0].note.content[:80]}\"")

        # Inference discount check
        inf_results = [r for r in await api.retrieve("outdoor activities", peer.id)
                       if r.note.note_type == "inference"]
        if inf_results:
            status(
                "Inference discount = 0.7",
                inf_results[0].inference_discount == 0.7,
                str(inf_results[0].inference_discount),
            )

        # Access recording
        note1_after = await api.get_note(note1.id)
        if note1_after and note1_after.access_count > 0:
            status(
                "Access recording works",
                True,
                f"access_count={note1_after.access_count}",
            )
        else:
            # note1 may not have been in the result set; check any returned note
            any_accessed = any(
                (await api.get_note(r.note.id)).access_count > 0
                for r in results
            )
            status("Access recording works", any_accessed)

        # Old search methods
        kw = await api.search_keyword("allergy", peer.id)
        vec = await api.search_vector("shellfish allergy", peer.id)
        hyb = await api.search_hybrid("allergy", peer.id)
        status(
            "Legacy search methods work",
            len(kw) > 0 and len(vec) > 0 and len(hyb) > 0,
            f"kw={len(kw)} vec={len(vec)} hyb={len(hyb)}",
        )

        await deriver.close()
        await api.close()

    print(f"\n{'='*50}")
    print("All checks passed.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    asyncio.run(main())
