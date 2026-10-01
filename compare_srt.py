import os
import re
import json
import requests
from difflib import SequenceMatcher
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "input"
OUTPUT_DIR = "output"
COMPARISON_DIR = "comparison"
MEMORY_DIR = os.path.join(COMPARISON_DIR, "memory")

MEMORY_JSONL = os.path.join(MEMORY_DIR, "translation_memory.jsonl")
MEMORY_JSON = os.path.join(MEMORY_DIR, "translation_memory.json")

COMPARISON_REPORT = os.path.join(COMPARISON_DIR, "comparison.txt")
COMPARISON_JSON = os.path.join(COMPARISON_DIR, "comparison.json")

OLLAMA_URL = "http://localhost:11434/api/chat"
MEMORY_MODEL = "qwen3:4b"

MEMORY_BATCH_SIZE = 20
OLLAMA_TIMEOUT = 120
OLLAMA_RETRIES = 3

MEMORY_CATEGORIES = {
    "character",
    "place",
    "verb",
    "object",
    "terminology",
    "expression",
    "scene",
    "description",
    "other",
}

# ============================================================
# SRT
# ============================================================

def parse_srt(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    result = {}

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        try:
            subtitle_id = int(lines[0].strip())
        except ValueError:
            continue

        result[subtitle_id] = {
            "id": subtitle_id,
            "timecode": lines[1].strip(),
            "text": "\n".join(lines[2:]).strip(),
        }

    return result


def find_srt_files(directory):
    if not os.path.isdir(directory):
        return []

    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(".srt")
    )


def select_file(directory, title):
    files = find_srt_files(directory)

    print("\n" + "=" * 80)
    print(title)
    print(f"Folder: {directory}/")
    print("=" * 80)

    if not files:
        print("ไม่พบไฟล์ .srt")
        return None

    for i, name in enumerate(files, 1):
        print(f"{i}. {name}")

    while True:
        choice = input("\nเลือกหมายเลขไฟล์: ").strip()

        if choice.lower() in {"q", "quit", "exit"}:
            return None

        if not choice.isdigit():
            print("กรุณาใส่หมายเลข")
            continue

        n = int(choice)

        if 1 <= n <= len(files):
            return os.path.join(directory, files[n - 1])

        print("หมายเลขไม่ถูกต้อง")


def normalize_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\\N", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# COMPARISON
# ============================================================

def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def compare_gold_ai(gold_data, ai_data):
    """
    เปรียบเทียบตาม subtitle ID
    ไม่แก้ไข Gold หรือ AI output
    """

    all_ids = sorted(set(gold_data) | set(ai_data))
    results = []

    for subtitle_id in all_ids:
        gold = gold_data.get(subtitle_id)
        ai = ai_data.get(subtitle_id)

        gold_text = gold["text"] if gold else ""
        ai_text = ai["text"] if ai else ""

        score = similarity(gold_text, ai_text)

        results.append({
            "id": subtitle_id,
            "timecode": (
                gold["timecode"] if gold
                else ai["timecode"] if ai
                else ""
            ),
            "gold": gold_text,
            "ai": ai_text,
            "similarity": round(score, 6),
            "gold_exists": gold is not None,
            "ai_exists": ai is not None,
        })

    return results


def calculate_overall_score(results):
    valid = [
        x for x in results
        if x["gold_exists"] and x["ai_exists"]
    ]

    if not valid:
        return 0.0

    return sum(x["similarity"] for x in valid) / len(valid)


# ============================================================
# OLLAMA
# ============================================================

MEMORY_SYSTEM_PROMPT = r"""
You build a Chinese -> Thai subtitle translation memory.

Your task is ONLY to extract reusable ATOMIC words or SHORT PHRASES
from SOURCE and their exact corresponding Thai wording from GOLD.

IMPORTANT:
1. SOURCE must be copied exactly from the Chinese SOURCE.
2. TARGET must be copied exactly from the Thai GOLD.
3. Never invent, rewrite, paraphrase, translate, normalize, or correct
   SOURCE or TARGET.
4. Do NOT store whole subtitle sentences.
5. Prefer individual words or short phrases.
6. A longer phrase is allowed only when it is a fixed expression,
   person's name, place name, title, terminology, or another unit whose
   meaning depends on the complete phrase.
7. Ignore generic particles and meaningless fragments.
8. Every SOURCE must be a contiguous substring of SOURCE.
9. Every TARGET must be a contiguous substring of GOLD.
10. Return category exactly as one of:
    character, place, verb, object, terminology, expression,
    scene, description, other.
11. Return JSON only.

Output:
{
  "items": [
    {
      "source": "中国原文中的词或短语",
      "target": "Gold中的泰文词或短语",
      "category": "terminology"
    }
  ]
}
"""


def call_ollama(system_prompt, user_prompt):
    payload = {
        "model": MEMORY_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 2048,
        },
        "keep_alive": "10m",
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()
    content = data.get("message", {}).get("content", "")

    if not content:
        raise ValueError(
            f"Ollama ไม่มี response: {data}"
        )

    return content.strip()


# ============================================================
# MEMORY
# ============================================================

def memory_key(source, target):
    return (
        normalize_text(source).casefold(),
        normalize_text(target).casefold(),
    )


def load_memory():
    memory = {}

    if not os.path.exists(MEMORY_JSONL):
        return memory

    with open(MEMORY_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = normalize_text(item.get("source", ""))
            target = normalize_text(item.get("target", ""))

            if not source or not target:
                continue

            category = str(
                item.get("category", "other")
            ).lower()

            if category not in MEMORY_CATEGORIES:
                category = "other"

            try:
                count = int(item.get("count", 1))
            except (TypeError, ValueError):
                count = 1

            key = memory_key(source, target)

            memory[key] = {
                "source": source,
                "target": target,
                "category": category,
                "count": max(count, 1),
            }

    return memory


def save_memory(memory):
    os.makedirs(MEMORY_DIR, exist_ok=True)

    items = list(memory.values())

    items.sort(
        key=lambda x: (
            x.get("category", "other"),
            x.get("source", ""),
            x.get("target", ""),
        )
    )

    # JSONL: ไม่มี id/index
    with open(MEMORY_JSONL, "w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                ) + "\n"
            )

    # JSON: ไม่มี id/index เช่นกัน
    by_category = {}

    for item in items:
        category = item["category"]
        by_category.setdefault(category, []).append(item)

    with open(MEMORY_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": by_category,
                "total": len(items),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def validate_atomic_item(item, source_text, gold_text):
    """
    ตรวจหลัง AI:
    - source ต้องอยู่ใน SOURCE
    - target ต้องอยู่ใน GOLD
    - ต้องไม่เป็นประโยคเต็มทั้งคู่
    """

    source = normalize_text(item.get("source", ""))
    target = normalize_text(item.get("target", ""))
    category = str(
        item.get("category", "other")
    ).strip().lower()

    source_normalized = normalize_text(source_text)
    gold_normalized = normalize_text(gold_text)

    if not source or not target:
        return None

    if source not in source_normalized:
        return None

    if target not in gold_normalized:
        return None

    # ห้ามเก็บทั้งประโยค
    if source == source_normalized:
        return None

    if target == gold_normalized:
        return None

    if category not in MEMORY_CATEGORIES:
        category = "other"

    return {
        "source": source,
        "target": target,
        "category": category,
    }


def extract_memory_batch(batch, batch_no, total_batches):
    """
    AI extract จาก Gold เท่านั้น
    ไม่ใช้ AI RAG เป็นข้อมูลสำหรับ Memory
    """

    prompt_items = []

    for item in batch:
        prompt_items.append({
            "source": item["source"],
            "gold": item["gold"],
        })

    user_prompt = json.dumps(
        {
            "task": "Extract atomic translation-memory units.",
            "items": prompt_items,
        },
        ensure_ascii=False,
        indent=2,
    )

    print(
        f"\n🧠 Memory batch "
        f"{batch_no}/{total_batches} "
        f"({len(batch)} subtitles)"
    )

    for attempt in range(1, OLLAMA_RETRIES + 1):
        try:
            raw = call_ollama(
                MEMORY_SYSTEM_PROMPT,
                user_prompt,
            )

            data = json.loads(raw)
            ai_items = data.get("items", [])

            if not isinstance(ai_items, list):
                raise ValueError(
                    "Ollama output items ไม่ใช่ list"
                )

            valid = []

            # AI output ไม่มี ID ที่จะนำไปเก็บใน Memory
            # จับคู่ตาม SOURCE/GOLD ของ batch โดยตรวจ substring
            for ai_item in ai_items:
                source = normalize_text(
                    ai_item.get("source", "")
                )
                target = normalize_text(
                    ai_item.get("target", "")
                )

                if not source or not target:
                    continue

                matched = False

                for original in batch:
                    if (
                        source in normalize_text(original["source"])
                        and
                        target in normalize_text(original["gold"])
                    ):
                        checked = validate_atomic_item(
                            ai_item,
                            original["source"],
                            original["gold"],
                        )

                        if checked:
                            valid.append(checked)
                            matched = True
                            break

                if not matched:
                    continue

            print(
                f"   ✅ AI extracted: {len(ai_items)}"
                f" | validated: {len(valid)}"
            )

            return valid

        except Exception as e:
            print(
                f"   ⚠️ Ollama error "
                f"(attempt {attempt}/{OLLAMA_RETRIES}): {e}"
            )

    print("   ❌ ข้าม batch นี้")
    return []


# ============================================================
# CONFLICT
# ============================================================

def build_memory_from_gold(gold_data, ai_data):
    """
    สำคัญ:
    - ใช้ Gold เท่านั้นในการสร้าง Memory
    - AI ใช้แค่เปรียบเทียบ ไม่แตะต้อง output
    - Conflict จะยังไม่ถามทันที
    - รวบรวมทั้งหมดก่อน แล้วถามตอนท้าย
    """

    memory = load_memory()

    all_gold = []

    for subtitle_id in sorted(gold_data):
        gold_item = gold_data[subtitle_id]

        all_gold.append({
            "id": subtitle_id,
            "source": gold_item["source"] if "source" in gold_item else "",
            "gold": gold_item["text"],
        })

    # SRT Gold ไม่มี source field เพราะเป็นไฟล์ภาษาไทย
    # ดังนั้น source จะถูกเติมจาก input/source ใน main
    return memory


def extract_gold_memory(source_data, gold_data):
    """
    สร้างรายการ SOURCE(Gold pair) สำหรับ AI extraction
    """

    pairs = []

    common_ids = sorted(
        set(source_data) & set(gold_data)
    )

    for subtitle_id in common_ids:
        source = normalize_text(
            source_data[subtitle_id]["text"]
        )
        gold = normalize_text(
            gold_data[subtitle_id]["text"]
        )

        if not source or not gold:
            continue

        pairs.append({
            "id": subtitle_id,
            "source": source,
            "gold": gold,
        })

    return pairs


def add_extracted_to_memory(memory, extracted):
    """
    เพิ่ม Memory แบบไม่ถาม Conflict ทันที

    Duplicate:
      source + target เหมือนกัน -> count + 1

    Conflict:
      source เหมือน แต่ target ต่างกัน
      -> ส่งเข้า conflict list
    """

    conflicts = []
    duplicate_count = 0
    added_count = 0

    # รวมรายการในรอบนี้ก่อน
    current_by_source = {}

    for item in extracted:
        source = item["source"]
        target = item["target"]
        category = item["category"]

        key = memory_key(source, target)

        if key in memory:
            memory[key]["count"] += 1
            duplicate_count += 1
            continue

        # ตรวจ source เดียวกันแต่ target ต่างกัน
        source_key = normalize_text(source).casefold()

        existing_same_source = [
            x for x in memory.values()
            if normalize_text(x["source"]).casefold()
            == source_key
            and normalize_text(x["target"]).casefold()
            != normalize_text(target).casefold()
        ]

        if source_key in current_by_source:
            for previous in current_by_source[source_key]:
                if memory_key(
                    previous["source"],
                    previous["target"],
                ) != key:
                    conflicts.append({
                        "source": source,
                        "old": previous["target"],
                        "new": target,
                        "old_category": previous["category"],
                        "new_category": category,
                    })

        if existing_same_source:
            for old in existing_same_source:
                conflicts.append({
                    "source": source,
                    "old": old["target"],
                    "new": target,
                    "old_category": old["category"],
                    "new_category": category,
                })

        # ยังเพิ่ม candidate ใหม่เข้า memory
        # แต่ conflict จะถูกแก้ภายหลังในขั้นตอนเดียว
        memory[key] = {
            "source": source,
            "target": target,
            "category": category,
            "count": 1,
        }

        current_by_source.setdefault(
            source_key, []
        ).append({
            "source": source,
            "target": target,
            "category": category,
        })

        added_count += 1

    return memory, conflicts, duplicate_count, added_count


def unique_conflicts(conflicts):
    result = []
    seen = set()

    for c in conflicts:
        key = (
            normalize_text(c["source"]).casefold(),
            normalize_text(c["old"]).casefold(),
            normalize_text(c["new"]).casefold(),
        )

        # A/B กับ B/A ถือเป็น conflict เดียวกัน
        reverse = (
            key[0],
            key[2],
            key[1],
        )

        if key in seen or reverse in seen:
            continue

        seen.add(key)
        result.append(c)

    return result


def resolve_conflicts(memory, conflicts):
    """
    หลัง extract ทั้งหมดแล้วจึงถาม Conflict ทีเดียว

    1 = เก่า
    2 = ใหม่
    3 = ทั้งคู่
    """

    conflicts = unique_conflicts(conflicts)

    if not conflicts:
        print("\n✅ ไม่พบ Conflict")
        return memory

    print("\n" + "=" * 80)
    print(f"⚠️ MEMORY CONFLICTS: {len(conflicts)}")
    print("=" * 80)

    for number, conflict in enumerate(conflicts, 1):
        source = conflict["source"]
        old = conflict["old"]
        new = conflict["new"]

        print("\n" + "-" * 80)
        print(f"Conflict {number}/{len(conflicts)}")
        print(f"Source : {source}")
        print(f"1 เก่า : {old}")
        print(f"2 ใหม่ : {new}")
        print("3 ทั้งคู่")

        while True:
            choice = input(
                "เลือก [1/2/3]: "
            ).strip()

            if choice in {"1", "2", "3"}:
                break

            print("กรุณาเลือก 1, 2 หรือ 3")

        old_key = memory_key(source, old)
        new_key = memory_key(source, new)

        if choice == "1":
            # เก็บเก่า, ลบใหม่
            if new_key in memory:
                del memory[new_key]

        elif choice == "2":
            # ลบเก่า, เก็บใหม่
            if old_key in memory:
                old_count = memory[old_key].get("count", 1)
            else:
                old_count = 0

            if new_key in memory:
                memory[new_key]["count"] += 1
            else:
                memory[new_key] = {
                    "source": source,
                    "target": new,
                    "category": conflict["new_category"],
                    "count": max(1, old_count),
                }

            if old_key in memory:
                del memory[old_key]

        elif choice == "3":
            # เก็บทั้งคู่ ไม่ทำอะไร
            pass

    return memory


# ============================================================
# REPORT
# ============================================================

def print_comparison(results):
    print("\n" + "=" * 80)
    print("GOLD vs AI RAG")
    print("=" * 80)

    for item in results:
        print(
            f"[{item['id']}] "
            f"{item['similarity']:.2%}"
        )

        print(f"  GOLD: {item['gold']}")
        print(f"  AI  : {item['ai']}")
        print()

    valid = [
        x for x in results
        if x["gold_exists"] and x["ai_exists"]
    ]

    overall = calculate_overall_score(results)

    print("-" * 80)
    print(f"เปรียบเทียบทั้งหมด: {len(results)}")
    print(f"มีทั้ง Gold และ AI: {len(valid)}")
    print(f"คะแนนรวม Gold ↔ AI: {overall:.2%}")


def save_comparison(results, gold_file, ai_file):
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    overall = calculate_overall_score(results)

    data = {
        "gold_file": os.path.basename(gold_file),
        "ai_file": os.path.basename(ai_file),
        "overall_similarity": round(overall, 6),
        "items": results,
    }

    with open(
        COMPARISON_JSON,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        COMPARISON_REPORT,
        "w",
        encoding="utf-8",
    ) as f:
        f.write("GOLD vs AI RAG COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"Gold: {gold_file}\n"
            f"AI:   {ai_file}\n"
        )
        f.write(
            f"Overall similarity: {overall:.2%}\n\n"
        )

        for item in results:
            f.write(
                f"[{item['id']}] "
                f"{item['similarity']:.2%}\n"
            )
            f.write(
                f"GOLD: {item['gold']}\n"
            )
            f.write(
                f"AI:   {item['ai']}\n\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("GOLD / AI COMPARATOR + TRANSLATION MEMORY")
    print("=" * 80)

    # 1. INPUT CHINESE
    source_file = select_file(
        INPUT_DIR,
        "1. เลือกไฟล์ INPUT ภาษาจีน",
    )

    if not source_file:
        return

    # 2. GOLD
    gold_file = select_file(
        OUTPUT_DIR,
        "2. เลือกไฟล์ OUTPUT - GOLD ที่คนแปลแล้ว",
    )

    if not gold_file:
        return

    # 3. AI RAG
    ai_file = select_file(
        OUTPUT_DIR,
        "3. เลือกไฟล์ OUTPUT - AI RAG",
    )

    if not ai_file:
        return

    print("\n" + "=" * 80)
    print("SELECTED FILES")
    print("=" * 80)

    print(f"INPUT : {source_file}")
    print(f"GOLD  : {gold_file}")
    print(f"AI RAG: {ai_file}")

    print("\nกำลังอ่านไฟล์...")

    source_data = parse_srt(source_file)
    gold_data = parse_srt(gold_file)
    ai_data = parse_srt(ai_file)

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    print("\nกำลังเปรียบเทียบ Gold กับ AI RAG...")

    results = compare_gold_ai(
        gold_data,
        ai_data,
    )

    print_comparison(results)

    save_comparison(
        results,
        gold_file,
        ai_file,
    )

    # --------------------------------------------------------
    # Memory extraction
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("BUILD TRANSLATION MEMORY FROM GOLD")
    print("=" * 80)

    pairs = extract_gold_memory(
        source_data,
        gold_data,
    )

    print(
        f"Gold pairs สำหรับ Memory: {len(pairs)}"
    )

    if not pairs:
        print(
            "❌ ไม่พบ SOURCE/GOLD ID ที่ตรงกัน"
        )
        return

    extracted = []

    total_batches = (
        len(pairs) + MEMORY_BATCH_SIZE - 1
    ) // MEMORY_BATCH_SIZE

    for start in range(
        0,
        len(pairs),
        MEMORY_BATCH_SIZE,
    ):
        batch = pairs[
            start:start + MEMORY_BATCH_SIZE
        ]

        batch_result = extract_memory_batch(
            batch,
            start // MEMORY_BATCH_SIZE + 1,
            total_batches,
        )

        extracted.extend(batch_result)

    print(
        f"\nAtomic Memory หลัง validation: "
        f"{len(extracted)}"
    )

    # --------------------------------------------------------
    # Duplicate + Conflict collection
    # --------------------------------------------------------

    memory = load_memory()

    (
        memory,
        conflicts,
        duplicate_count,
        added_count,
    ) = add_extracted_to_memory(
        memory,
        extracted,
    )

    print("\n" + "-" * 80)
    print(f"Memory ใหม่: {added_count}")
    print(f"Duplicate: {duplicate_count}")
    print(
        f"Conflict ที่รวบรวมไว้: "
        f"{len(unique_conflicts(conflicts))}"
    )

    # --------------------------------------------------------
    # Resolve ALL conflicts only after extraction is complete
    # --------------------------------------------------------

    memory = resolve_conflicts(
        memory,
        conflicts,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_memory(memory)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Memory total: {len(memory)}"
    )
    print(
        f"Memory JSONL: {MEMORY_JSONL}"
    )
    print(
        f"Memory JSON:  {MEMORY_JSON}"
    )
    print(
        f"Comparison:   {COMPARISON_JSON}"
    )
    print(
        f"Report:       {COMPARISON_REPORT}"
    )

    print("\n⚠️ AI RAG output ไม่ถูกแก้ไข")


if __name__ == "__main__":
    main()
