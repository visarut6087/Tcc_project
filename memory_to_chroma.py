# -*- coding: utf-8 -*-
"""
เก็บ Translation Memory (จาก build_memory.py) เข้า ChromaDB แบบ vector
เพื่ อนำไปเรียกใช้ใน translator.py รอบถัดไป

Embedding: Ollama local (nomic-embed-text)  -> 1 ตัวถอก = 1 vector (768 dim)
Storage   : ChromaDB persistent (cosine) ใน chroma_db/

ใช้:
    python memory_to_chroma.py                       # import ทุก entry (idempotent)
    python memory_to_chroma.py --min-count 3         # เฉพาะ Gold (พบ >= 3 ครั้ ง)
    python memory_to_chroma.py --limit 500           # ทดสอบเฉพาะ 500 แรก
    python memory_to_chroma.py --rebuild             # ล้าง collection แล้วสร้างใหม่
    python memory_to_chroma.py --status              # ดูสถานะ + ทดสอบคั้นหา
    python memory_to_chroma.py --test " texto zh "   # ทดสอบ semantic search
"""

import os
import sys
import json
import hashlib
import argparse

import requests
import chromadb

# ============================================================
# CONFIG
# ============================================================

MEMORY_FILE = os.path.join(
    "comparison", "memory", "translation_memory.jsonl"
)

CHROMA_DIR = "chroma_db"
COLLECTION = "translation_memory"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

EMBED_BATCH = 32
QUERY_BATCH = 32
TIMEOUT = 600

# field ที่ Chroma ไม่รองรบั (ตองเป็น str/int/float/bool)
LIST_FIELDS = ("zh_words", "th_words")


# ============================================================
# LOAD MEMORY
# ============================================================

def load_memory(path, min_count=1, limit=0):

    if not os.path.exists(path):
        print(f"❌ not found: {path}")
        print("   รัน `python build_memory.py` ก่อน")
        sys.exit(1)

    entries = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = str(
                item.get("source", item.get("source_zh", ""))
            ).strip()

            target = str(
                item.get("target", item.get("target_th", ""))
            ).strip()

            if not source or not target:
                continue

            try:
                count = int(item.get("count", 1))
            except (TypeError, ValueError):
                count = 1

            if count < min_count:
                continue

            entries.append(
                {
                    "source": source,
                    "target": target,
                    "count": count,
                    "category": item.get("category", "other"),
                    "keywords": item.get("keywords", {}),
                }
            )

            if limit and len(entries) >= limit:
                break

    return entries


def make_document(entry):

    kw = entry.get("keywords") or {}

    kw_zh = str(kw.get("zh", "")).strip()
    kw_th = str(kw.get("th", "")).strip()

    # ใส้ tัง้ ประโยค + keywords เพื่ อให้ semantic ตรงกว่าน
    return (
        f"ZH: {entry['source']}\n"
        f"TH: {entry['target']}\n"
        f"KEYWORDS: {kw_zh} | {kw_th}"
    )


def make_id(entry):

    key = f"{entry['source']}||{entry['target']}"

    return hashlib.sha1(
        key.encode("utf-8")
    ).hexdigest()


def make_metadata(entry):

    kw = entry.get("keywords") or {}

    meta = {
        "source": entry["source"],
        "target": entry["target"],
        "count": int(entry.get("count", 1)),
        "category": str(entry.get("category", "other")),
        "kw_zh": str(kw.get("zh", "")),
        "kw_th": str(kw.get("th", "")),
    }

    # กัน field ที่เป็น list/null (Chroma จะปฏเสธ)
    clean = {}

    for k, v in meta.items():
        if isinstance(v, (list, tuple, dict)):
            v = " ".join(str(x) for x in v)
        if v is None:
            v = ""
        clean[k] = v

    return clean




# ============================================================
# EMBEDDING (Ollama local)
# ============================================================

def embed_texts(texts, url=OLLAMA_URL, model=EMBED_MODEL):

    vectors = []

    for start in range(0, len(texts), EMBED_BATCH):

        batch = texts[start:start + EMBED_BATCH]

        response = requests.post(
            url.rstrip("/") + "/api/embed",
            json={"model": model, "input": batch},
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings or len(embeddings) != len(batch):
            raise RuntimeError(
                "Ollama ไม่ได้ embedding ครบทุกต้ว "
                f"(ได้ {len(embeddings or [])} จาก {len(batch)}) "
                "- ตรวจชื่ อ embedding model / pull model แล้วยัง"
            )

        vectors.extend(embeddings)

        done = min(start + EMBED_BATCH, len(texts))

        print(
            f"\r   ⚙ embedding {done}/{len(texts)}",
            end="",
            flush=True,
        )

    if texts:
        print()

    return vectors


# ============================================================
# COLLECTION
# ============================================================

def get_collection(client, name=COLLECTION, rebuild=False):

    if rebuild:
        try:
            client.delete_collection(name)
            print(f"🗑 deleted existing collection: {name}")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "description": "Translation Memory (zh->th) vectors for translator.py",
            "embedding_model": EMBED_MODEL,
        },
    )


# ============================================================
# IMPORT
# ============================================================

def import_memory(col, entries):

    total = col.count()

    existing = set()

    if total:
        # ดึง id เดิมทัง้ หมดมาเทียบ (idempotent / resume ได้)
        got = col.get(include=[])
        existing = set(got.get("ids", []))
        print(f"♻ existing vectors in collection: {total}")

    todo = [
        e
        for e in entries
        if make_id(e) not in existing
    ]

    print(
        f"📊 memory entries: {len(entries)} | "
        f"new to embed: {len(todo)}"
    )

    if not todo:
        print("✅ Everything already imported - nothing to do")
        return 0

    documents = [make_document(e) for e in todo]
    metadatas = [make_metadata(e) for e in todo]
    ids = [make_id(e) for e in todo]

    vectors = embed_texts(documents)

    chunk = EMBED_BATCH * 4

    for start in range(0, len(todo), chunk):

        end = start + chunk

        col.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=vectors[start:end],
        )

        print(f"   💾 saved {min(end, len(todo))}/{len(todo)}")

    return len(todo)


# ============================================================
# SEARCH (ไว้ทดสอบ / ไว้ translator.py นำไปใช้)
# ============================================================

def search(col, query, top_k=5):

    qv = embed_texts([query])

    res = col.query(
        query_embeddings=qv,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    out = []

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for i in range(len(ids)):

        meta = metas[i] or {}

        out.append(
            {
                "id": ids[i],
                "source": meta.get("source", ""),
                "target": meta.get("target", ""),
                "similarity": round(1.0 - float(dists[i]), 4),
                "count": meta.get("count", 1),
                "category": meta.get("category", "other"),
                "document": docs[i],
            }
        )

    return out


# ============================================================
# STATUS
# ============================================================

def show_status(col, query=None):

    print("=" * 50)
    print(f"Collection : {col.name}")
    print(f"Vectors    : {col.count()}")
    print(f"Metadata   : {col.metadata}")
    print("=" * 50)

    sample = col.peek(3)

    for i, d in enumerate(sample.get("documents", []) or []):
        m = (sample.get("metadatas") or [{}])[i] or {}
        print(
            f"\n[{i + 1}] {m.get('source')} -> {m.get('target')} "
            f"(count={m.get('count')}, cat={m.get('category')})"
        )
        print(f"    {d.splitlines()[0]}")

    if query:
        print("\n" + "-" * 50)
        print(f"TEST query: {query}")
        for hit in search(col, query, top_k=5):
            print(
                f"  {hit['similarity']:.4f} | "
                f"{hit['source']} -> {hit['target']} "
                f"(count={hit['count']})"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    global OLLAMA_URL, EMBED_MODEL

    ap = argparse.ArgumentParser(
        description="Import Translation Memory into ChromaDB vectors."
    )

    ap.add_argument("--memory-file", default=MEMORY_FILE)
    ap.add_argument("--chroma-dir", default=CHROMA_DIR)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--embedding", default=EMBED_MODEL)
    ap.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="เก็บเฉพาะ entry ที่พบ >= N ครั้ง (3 = Gold)",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="ลบ collection เดิมก่อนแล้วสร้างใหม่",
    )
    ap.add_argument(
        "--status",
        action="store_true",
        help="แสดงสถานะ collection",
    )
    ap.add_argument(
        "--test",
        default="",
        help="ทดสอบ semantic search ด้วยข้อความจีน",
    )

    args = ap.parse_args()

    OLLAMA_URL = args.ollama_url
    EMBED_MODEL = args.embedding

    client = chromadb.PersistentClient(path=args.chroma_dir)
    col = get_collection(client, args.collection, rebuild=args.rebuild)

    # โหมดทดสอบ / ดดู สถานะ (ไมตอง import)
    if args.test and col.count():
        print(f"TEST query: {args.test}\n")
        for hit in search(col, args.test, top_k=5):
            print(
                f"  {hit['similarity']:.4f} | "
                f"{hit['source']} -> {hit['target']} "
                f"(count={hit['count']}, cat={hit['category']})"
            )
        return

    if args.status:
        show_status(col)
        return

    entries = load_memory(
        args.memory_file,
        min_count=args.min_count,
        limit=args.limit,
    )

    print("=" * 50)
    print("IMPORT TRANSLATION MEMORY -> CHROMA")
    print(f"  memory    : {args.memory_file}")
    print(f"  chroma    : {args.chroma_dir}/{args.collection}")
    print(f"  embedding : {EMBED_MODEL} @ {OLLAMA_URL}")
    print(f"  min-count : {args.min_count}")
    print("=" * 50)

    added = import_memory(col, entries)

    print(f"\n✅ added: {added}")
    print(f"   total vectors now: {col.count()}")

    # ทดสอบให้อัตโนมัติ 1 query
    if col.count():
        probe = entries[0]["source"] if entries else "你好"
        print(f"\n--- sample search: {probe} ---")
        for hit in search(col, probe, top_k=3):
            print(
                f"  {hit['similarity']:.4f} | "
                f"{hit['source']} -> {hit['target']} "
                f"(count={hit['count']})"
            )


if __name__ == "__main__":

    main()

