#!/usr/bin/env python3
"""
Extract useful Chinese words/phrases from Chinese SRT, align optional Thai SRT,
use a local Ollama model to translate/extract reusable expressions, write review JSON,
and store records in ChromaDB.

Requirements:
  pip install requests srt chromadb

Ollama setup (examples):
  ollama pull qwen3:8b
  ollama pull nomic-embed-text

Example:
  python srt_to_chroma_local.py --zh "data/movie.zh.srt" --th "data/movie.th.srt"
  python srt_to_chroma_local.py --zh "data/movie.zh.srt" --out-json "review.json" --collection "chinese_phrases"

If --th is omitted, the script asks the local LLM to translate the Chinese subtitle.
"""

import argparse
import json
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import Any

import requests
import srt
import chromadb


def parse_srt(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    items = []
    for i, sub in enumerate(srt.parse(text)):
        content = re.sub(r"<[^>]+>", "", sub.content)
        content = re.sub(r"\{\\an\d+\}", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            continue
        items.append({
            "index": i,
            "start": sub.start.total_seconds(),
            "end": sub.end.total_seconds(),
            "text": content,
        })
    return items


def nearest_translation(zh_item: dict, th_items: list[dict], tolerance: float = 2.5) -> str:
    """Match by time overlap/proximity rather than assuming subtitle indices match."""
    if not th_items:
        return ""
    z_mid = (zh_item["start"] + zh_item["end"]) / 2
    ranked = []
    for t in th_items:
        overlap = max(0.0, min(zh_item["end"], t["end"]) - max(zh_item["start"], t["start"]))
        t_mid = (t["start"] + t["end"]) / 2
        distance = abs(z_mid - t_mid)
        ranked.append((overlap, distance, t["text"]))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    best = ranked[0]
    return best[2] if best[0] > 0 or best[1] <= tolerance else ""


def ollama_chat(base_url: str, model: str, prompt: str, timeout: int) -> str:
    response = requests.post(
        base_url.rstrip("/") + "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": (
                    "You are a careful Chinese-learning assistant. Use only the supplied subtitle context. "
                    "Do not invent source sentences or claim unsupported meanings. Return valid JSON only."
                )},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def translate_batch(
    base_url: str, model: str, rows: list[dict], timeout: int
) -> dict[int, str]:
    """Translate Chinese rows locally if no Thai subtitle match is available."""
    missing = [r for r in rows if not r.get("thai")]
    if not missing:
        return {}
    payload = [{"id": r["id"], "zh": r["zh"]} for r in missing]
    prompt = (
        "Translate each Chinese subtitle into natural Thai. Preserve meaning and tone. "
        "Do not add explanations. Return JSON object with key 'translations', an array of "
        "objects {id, th}. Input:\n" + json.dumps(payload, ensure_ascii=False)
    )
    data = json.loads(ollama_chat(base_url, model, prompt, timeout))
    return {int(x["id"]): str(x["th"]).strip() for x in data.get("translations", [])}


def extract_batch(
    base_url: str, model: str, rows: list[dict], timeout: int
) -> list[dict]:
    """
    For each subtitle target, extract only useful reusable words/phrases.
    Context includes previous/current/next aligned subtitle pairs.
    """
    compact = []
    for r in rows:
        compact.append({
            "id": r["id"],
            "context_before": r.get("before", {}),
            "target": {"zh": r["zh"], "th": r.get("thai", "")},
            "context_after": r.get("after", {}),
        })

    prompt = r"""
Task: extract useful Chinese vocabulary and reusable spoken phrases from bilingual subtitle context.
Prioritize:
- pronouns/address forms + verbs or common sentence patterns;
- verbs, verb-object phrases, collocations, discourse phrases;
- expressions reusable in multiple real-life situations;
- context-sensitive words whose meaning changes with context.
Do not output every ordinary word. Avoid isolated particles unless they are useful.
Use the Thai translation and neighboring sentences as evidence, but do not assume word-for-word alignment.
Every item must be grounded in the target Chinese sentence. Keep phrase length useful (usually 1-8 Chinese characters, but longer natural expressions are allowed).
For each target id, return 0-5 items. Avoid duplicates within the same target.

Return JSON only with this exact shape:
{
  "items": [
    {
      "source_id": 1,
      "term_zh": "Chinese word or phrase",
      "pinyin": "pinyin if confident, otherwise empty string",
      "meaning_th": "Thai meaning in this context",
      "pattern": "reusable pattern such as 你 + 想 + V; empty if not applicable",
      "usage_th": "brief Thai note about when/how it is used",
      "register": "neutral|casual|polite|slang|vulgar|formal",
      "usefulness": 1,
      "confidence": 0.0,
      "evidence_zh": "exact substring or phrase from the target sentence",
      "evidence_th": "corresponding Thai subtitle, if available",
      "example_zh": "source sentence",
      "example_th": "Thai translation of source sentence"
    }
  ]
}
Use usefulness from 1 to 5. confidence from 0 to 1. Do not fabricate an example or add facts not supported by context. If uncertain, lower confidence. For Thai wording, use natural Thai. Input:
""" + json.dumps(compact, ensure_ascii=False)

    raw = ollama_chat(base_url, model, prompt, timeout)
    data = json.loads(raw)
    results = []
    valid_ids = {r["id"] for r in rows}
    for item in data.get("items", []):
        try:
            sid = int(item["source_id"])
            if sid not in valid_ids:
                continue
            term = str(item.get("term_zh", "")).strip()
            meaning = str(item.get("meaning_th", "")).strip()
            if not term or not meaning:
                continue
            item["source_id"] = sid
            item["usefulness"] = max(1, min(5, int(item.get("usefulness", 3))))
            item["confidence"] = max(0.0, min(1.0, float(item.get("confidence", 0.5))))
            results.append(item)
        except (KeyError, TypeError, ValueError):
            continue
    return results


def embed_texts(base_url: str, model: str, texts: list[str], timeout: int) -> list[list[float]]:
    response = requests.post(
        base_url.rstrip("/") + "/api/embed",
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama did not return embeddings for all texts. Check the embedding model name/version."
        )
    return embeddings


def main():
    ap = argparse.ArgumentParser(description="Extract Chinese words/phrases from SRT into ChromaDB.")
    ap.add_argument("--zh", required=True, help="Path to Chinese SRT")
    ap.add_argument("--th", default="", help="Optional Thai-translated SRT")
    ap.add_argument("--out-json", default="srt_vocab_review.json", help="JSON file for review")
    ap.add_argument("--chroma-dir", default="./chroma_db", help="Persistent ChromaDB directory")
    ap.add_argument("--collection", default="chinese_subtitle_phrases", help="Chroma collection name")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--llm", "--model", dest="llm", default="qwen3:8b", help="Local Ollama chat model, e.g. qwen3:8b, qwen3:14b")
    ap.add_argument("--embedding", "--embedding-model", dest="embedding", default="nomic-embed-text", help="Local Ollama embedding model")
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of Chinese subtitle entries to process; 0 means all")
    ap.add_argument("--batch-size", type=int, default=6, help="Subtitle targets per LLM extraction batch")
    ap.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds")
    ap.add_argument("--min-confidence", type=float, default=0.45, help="Minimum confidence to store")
    ap.add_argument("--min-usefulness", type=int, default=2, help="Minimum usefulness score to store")
    args = ap.parse_args()

    zh_items = parse_srt(args.zh)
    if not zh_items:
        sys.exit("No Chinese subtitle lines found.")
    total_zh_items = len(zh_items)
    if args.limit < 0:
        sys.exit("--limit must be 0 or greater (0 means process all subtitles).")
    if args.limit > 0:
        zh_items = zh_items[:args.limit]
    print(f"Chinese subtitle entries selected: {len(zh_items)}/{total_zh_items}")
    th_items = parse_srt(args.th) if args.th else []

    rows = []
    for i, z in enumerate(zh_items):
        rows.append({
            "id": i,
            "start": z["start"],
            "end": z["end"],
            "zh": z["text"],
            "thai": nearest_translation(z, th_items) if th_items else "",
        })

    # Fill gaps with local translation. This also covers unmatched lines in the Thai SRT.
    missing_rows = [r for r in rows if not r["thai"]]
    for start in range(0, len(missing_rows), args.batch_size):
        batch = missing_rows[start:start + args.batch_size]
        translations = translate_batch(args.ollama_url, args.llm, batch, args.timeout)
        for r in batch:
            r["thai"] = translations.get(r["id"], "")
        print(f"Translated missing lines: {min(start + args.batch_size, len(missing_rows))}/{len(missing_rows)}")

    # Attach neighboring subtitle context, preserving the bilingual pair.
    for i, r in enumerate(rows):
        r["before"] = (
            {"zh": rows[i - 1]["zh"], "th": rows[i - 1]["thai"], "id": i - 1}
            if i > 0 else {}
        )
        r["after"] = (
            {"zh": rows[i + 1]["zh"], "th": rows[i + 1]["thai"], "id": i + 1}
            if i + 1 < len(rows) else {}
        )

    all_items = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        try:
            found = extract_batch(args.ollama_url, args.llm, batch, args.timeout)
            all_items.extend(found)
        except Exception as e:
            print(f"WARNING: extraction batch starting at {start} failed: {e}", file=sys.stderr)
        print(f"Extracted subtitle lines: {min(start + args.batch_size, len(rows))}/{len(rows)}")

    # Stable IDs; keep duplicate terms if they carry distinct contextual evidence, but skip exact duplicate records.
    unique = {}
    for item in all_items:
        if item["confidence"] < args.min_confidence or item["usefulness"] < args.min_usefulness:
            continue
        source = rows[item["source_id"]]
        before = source.get("before", {})
        after = source.get("after", {})
        start_s = int(source["start"])
        record = {
            "term_zh": item.get("term_zh", ""),
            "pinyin": item.get("pinyin", ""),
            "meaning_th": item.get("meaning_th", ""),
            "pattern": item.get("pattern", ""),
            "usage_th": item.get("usage_th", ""),
            "register": item.get("register", "neutral"),
            "usefulness": item["usefulness"],
            "confidence": item["confidence"],
            "evidence_zh": item.get("evidence_zh", source["zh"]),
            "evidence_th": item.get("evidence_th", source["thai"]),
            "example_zh": item.get("example_zh", source["zh"]),
            "example_th": item.get("example_th", source["thai"]),
            "context_before_zh": before.get("zh", ""),
            "context_before_th": before.get("th", ""),
            "context_after_zh": after.get("zh", ""),
            "context_after_th": after.get("th", ""),
            "source_file": Path(args.zh).name,
            "source_id": item["source_id"],
            "start_seconds": source["start"],
            "end_seconds": source["end"],
        }
        # Include context in the dedupe key so different meanings/usages can survive.
        key = hashlib.sha1(
            (record["term_zh"] + "|" + record["meaning_th"] + "|" + record["evidence_zh"]).encode("utf-8")
        ).hexdigest()
        unique.setdefault(key, record)

    records = list(unique.values())
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote review JSON: {out_path.resolve()} ({len(records)} records)")

    if not records:
        print("No records passed the filters; JSON was written as an empty list.")
        return

    # Embed a retrieval-friendly text containing the term, Thai meaning, pattern and examples.
    embed_inputs = []
    for r in records:
        embed_inputs.append(
            f"Chinese term: {r['term_zh']}\nPinyin: {r['pinyin']}\n"
            f"Thai meaning: {r['meaning_th']}\nPattern: {r['pattern']}\n"
            f"Usage: {r['usage_th']}\nRegister: {r['register']}\n"
            f"Chinese example: {r['example_zh']}\nThai example: {r['example_th']}\n"
            f"Previous context: {r['context_before_zh']} / {r['context_before_th']}\n"
            f"Next context: {r['context_after_zh']} / {r['context_after_th']}"
        )

    print(f"Creating embeddings locally with {args.embedding}...")
    embeddings = []
    for start in range(0, len(embed_inputs), 32):
        embeddings.extend(embed_texts(args.ollama_url, args.embedding, embed_inputs[start:start + 32], args.timeout))

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "description": "Chinese-Thai subtitle vocabulary and reusable phrases"},
    )

    ids, docs, metas = [], [], []
    for i, (record, doc) in enumerate(zip(records, embed_inputs)):
        stable = hashlib.sha1(
            (record["source_file"] + "|" + record["term_zh"] + "|" +
             record["meaning_th"] + "|" + record["evidence_zh"] + "|" +
             str(record["start_seconds"])).encode("utf-8")
        ).hexdigest()
        ids.append(stable)
        docs.append(doc)
        # Chroma metadata must be scalar values, not nested objects or nulls.
        metas.append({
            "term_zh": record["term_zh"],
            "pinyin": record["pinyin"],
            "meaning_th": record["meaning_th"],
            "pattern": record["pattern"],
            "usage_th": record["usage_th"],
            "register": record["register"],
            "usefulness": record["usefulness"],
            "confidence": record["confidence"],
            "evidence_zh": record["evidence_zh"],
            "evidence_th": record["evidence_th"],
            "example_zh": record["example_zh"],
            "example_th": record["example_th"],
            "context_before_zh": record["context_before_zh"],
            "context_before_th": record["context_before_th"],
            "context_after_zh": record["context_after_zh"],
            "context_after_th": record["context_after_th"],
            "source_file": record["source_file"],
            "source_id": record["source_id"],
            "start_seconds": float(record["start_seconds"]),
            "end_seconds": float(record["end_seconds"]),
        })

    # Upsert allows re-running without duplicate IDs.
    for start in range(0, len(ids), 64):
        collection.upsert(
            ids=ids[start:start + 64],
            documents=docs[start:start + 64],
            metadatas=metas[start:start + 64],
            embeddings=embeddings[start:start + 64],
        )
    print(f"Stored {len(records)} records in ChromaDB collection '{args.collection}' at {Path(args.chroma_dir).resolve()}")
    print("Tip: inspect the JSON first; LLM extraction is probabilistic and may need manual review.")


if __name__ == "__main__":
    main()
