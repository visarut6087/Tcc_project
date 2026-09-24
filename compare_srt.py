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
REPORT_DIR = "comparison"

TRAINING_JSONL = os.path.join(REPORT_DIR, "translation_training.jsonl")
TRAINING_JSON = os.path.join(REPORT_DIR, "translation_training.json")
FEEDBACK_JSON = os.path.join(REPORT_DIR, "translation_feedback.json")
TRAINING_DB_JSONL = os.path.join(REPORT_DIR, "translation_training_db.jsonl")

# Translation Memory
MEMORY_DIR = os.path.join(REPORT_DIR, "memory")
MEMORY_FILE = os.path.join(MEMORY_DIR, "translation_memory.jsonl")
MEMORY_INDEX_FILE = os.path.join(MEMORY_DIR, "translation_memory.json")

# ใช้ AI เฉพาะตอน "จัดหมวดหมู่" Memory ใหม่
OLLAMA_URL = "http://localhost:11434/api/chat"
MEMORY_MODEL = "qwen3:4b"
MEMORY_BATCH_SIZE = 20
MEMORY_TIMEOUT = 60
MEMORY_RETRIES = 2

MEMORY_CATEGORIES = {
    "character",
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
    subtitles = {}

    for block in blocks:
        lines = block.strip().splitlines()

        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        subtitles[index] = {
            "index": index,
            "timecode": lines[1].strip(),
            "text": "\n".join(lines[2:]).strip(),
        }

    return subtitles


def find_srt_files(directory):
    if not os.path.exists(directory):
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
        print(f"ไม่พบไฟล์ SRT ใน {directory}/")
        return None

    for i, filename in enumerate(files, 1):
        print(f"{i}. {filename}")

    while True:
        choice = input("\nเลือกหมายเลขไฟล์: ").strip()

        if not choice.isdigit():
            print("กรุณาใส่หมายเลข")
            continue

        number = int(choice)

        if 1 <= number <= len(files):
            return os.path.join(directory, files[number - 1])

        print("หมายเลขไม่ถูกต้อง")


def normalize_text(text):
    if not text:
        return ""

    text = text.replace("\\N", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def similarity(text1, text2):
    a = normalize_text(text1)
    b = normalize_text(text2)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ============================================================
# COMPARISON
# ============================================================

def compare_srt(source, gold, rag):
    source_data = parse_srt(source)
    gold_data = parse_srt(gold)
    rag_data = parse_srt(rag)

    all_ids = sorted(
        set(source_data) |
        set(gold_data) |
        set(rag_data)
    )

    results = []

    for index in all_ids:
        source_item = source_data.get(index)
        gold_item = gold_data.get(index)
        rag_item = rag_data.get(index)

        source_text = source_item["text"] if source_item else ""
        gold_text = gold_item["text"] if gold_item else ""
        rag_text = rag_item["text"] if rag_item else ""

        results.append({
            "index": index,
            "timecode": (
                source_item["timecode"] if source_item
                else gold_item["timecode"] if gold_item
                else rag_item["timecode"] if rag_item
                else ""
            ),
            "source": source_text,
            "gold": gold_text,
            "rag": rag_text,
            "gold_similarity_to_source": similarity(
                source_text, gold_text
            ),
            "rag_similarity_to_source": similarity(
                source_text, rag_text
            ),
            "gold_vs_rag": similarity(
                gold_text, rag_text
            ),
            "same": (
                normalize_text(gold_text) ==
                normalize_text(rag_text)
            ),
            "source_exists": source_item is not None,
            "gold_exists": gold_item is not None,
            "rag_exists": rag_item is not None,
        })

    return results


# ============================================================
# TRAINING DATA
# ============================================================

def training_key(example):
    source = normalize_text(
        example.get("source_zh", "")
    )
    target = normalize_text(
        example.get("target_th", "")
    )

    return (
        source.casefold(),
        target.casefold()
    )


def load_existing_training_keys():
    keys = set()

    if not os.path.exists(TRAINING_DB_JSONL):
        return keys

    with open(
        TRAINING_DB_JSONL,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                key = training_key(item)

                if key[0] and key[1]:
                    keys.add(key)

            except json.JSONDecodeError:
                continue

    return keys


def append_new_training_examples(examples):
    existing_keys = load_existing_training_keys()

    new_examples = []
    duplicate_count = 0

    for example in examples:
        key = training_key(example)

        if not key[0] or not key[1]:
            continue

        if key in existing_keys:
            duplicate_count += 1
            continue

        existing_keys.add(key)
        new_examples.append(example)

    if new_examples:
        with open(
            TRAINING_DB_JSONL,
            "a",
            encoding="utf-8"
        ) as f:
            for example in new_examples:
                f.write(
                    json.dumps(
                        example,
                        ensure_ascii=False
                    ) + "\n"
                )

    return new_examples, duplicate_count


def deduplicate_examples(examples):
    seen = set()
    unique = []
    duplicate_count = 0

    for example in examples:
        key = training_key(example)

        if key in seen:
            duplicate_count += 1
            continue

        seen.add(key)
        unique.append(example)

    return unique, duplicate_count


def build_training_examples(
    results,
    source_file,
    gold_file,
    rag_file
):
    examples = []

    for item in results:
        if not item["source_exists"]:
            continue

        if not item["gold_exists"]:
            continue

        source = item["source"].strip()
        gold = item["gold"].strip()
        rag = item["rag"].strip()

        if not source or not gold:
            continue

        if item["rag_exists"] and item["same"]:
            example_type = "approved"
        elif item["rag_exists"]:
            example_type = "corrected"
        else:
            example_type = "missing_output"

        examples.append({
            "id": item["index"],
            "source_zh": source,
            "target_th": gold,
            "previous_rag": rag,
            "example_type": example_type,
            "previous_similarity": round(
                item["gold_vs_rag"], 4
            ),
            "source_to_gold_similarity": round(
                item["gold_similarity_to_source"], 4
            ),
            "timecode": item["timecode"],
            "source_file": os.path.basename(source_file),
            "gold_file": os.path.basename(gold_file),
            "rag_file": os.path.basename(rag_file),
        })

    return examples


def export_training_data(
    results,
    source_file,
    gold_file,
    rag_file
):
    os.makedirs(REPORT_DIR, exist_ok=True)

    examples = build_training_examples(
        results,
        source_file,
        gold_file,
        rag_file
    )

    examples, duplicate_in_run = deduplicate_examples(
        examples
    )

    new_examples, duplicate_from_history = (
        append_new_training_examples(examples)
    )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    dataset = {
        "version": "1.0",
        "created_at": timestamp,
        "purpose": (
            "Translation memory / feedback dataset "
            "for next-round Chinese-to-Thai subtitle translation"
        ),
        "source_file": os.path.basename(source_file),
        "gold_file": os.path.basename(gold_file),
        "rag_file": os.path.basename(rag_file),
        "total_examples": len(new_examples),
        "duplicates_in_current_run": duplicate_in_run,
        "duplicates_from_previous_runs": (
            duplicate_from_history
        ),
        "examples": new_examples,
    }

    with open(
        TRAINING_JSONL,
        "w",
        encoding="utf-8"
    ) as f:
        for example in new_examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                ) + "\n"
            )

    with open(
        TRAINING_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2
        )

    corrected = [
        x for x in new_examples
        if x["example_type"] == "corrected"
    ]

    approved = [
        x for x in new_examples
        if x["example_type"] == "approved"
    ]

    feedback = {
        "version": "1.0",
        "created_at": timestamp,
        "statistics": {
            "total": len(examples),
            "approved": len(approved),
            "corrected": len(corrected),
            "missing_output": sum(
                x["example_type"] == "missing_output"
                for x in examples
            ),
        },
        "corrections": [
            {
                "source_zh": x["source_zh"],
                "wrong_translation": x["previous_rag"],
                "correct_translation": x["target_th"],
                "similarity": x["previous_similarity"],
                "id": x["id"],
            }
            for x in corrected
        ],
        "approved_examples": [
            {
                "source_zh": x["source_zh"],
                "translation": x["target_th"],
                "id": x["id"],
            }
            for x in approved
        ],
    }

    with open(
        FEEDBACK_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            feedback,
            f,
            ensure_ascii=False,
            indent=2
        )

    return (
        new_examples,
        duplicate_in_run,
        duplicate_from_history
    )


# ============================================================
# SIMPLE TRANSLATION MEMORY
# ============================================================

def memory_key(source, target):
    return (
        normalize_text(source).casefold(),
        normalize_text(target).casefold()
    )


def load_memory():
    memory = {}

    if not os.path.exists(MEMORY_FILE):
        return memory

    with open(
        MEMORY_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = normalize_text(
                item.get("source", "")
            )
            target = normalize_text(
                item.get("target", "")
            )

            if not source or not target:
                continue

            memory[memory_key(source, target)] = item

    return memory


def save_memory(memory):
    os.makedirs(MEMORY_DIR, exist_ok=True)

    items = list(memory.values())

    items.sort(
        key=lambda x: (
            x.get("count", 0),
            x.get("source", "")
        ),
        reverse=True
    )

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        for item in items:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    # ทำ index แบบอ่านง่ายสำหรับ debugging / GUI
    by_category = {}

    for item in items:
        category = item.get(
            "category",
            "other"
        )

        by_category.setdefault(
            category,
            []
        ).append(item)

    with open(
        MEMORY_INDEX_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "total": len(items),
                "categories": by_category,
            },
            f,
            ensure_ascii=False,
            indent=2
        )


def is_memory_candidate(source, target, example_type):
    """
    ใช้ข้อมูล Training DB ทั้งหมดเป็น Translation Memory

    ไม่กรองตามความยาวหรือ example_type
    เพราะ Gold เป็นข้อมูลที่ผ่านการยอมรับแล้ว

    AI มีหน้าที่ "จัดหมวดหมู่" เท่านั้น
    ไม่ได้สร้างหรือแก้คำแปล
    """
    source = normalize_text(source)
    target = normalize_text(target)

    return bool(source and target)


# ============================================================
# AI CATEGORY CLASSIFIER
# ============================================================

MEMORY_SYSTEM_PROMPT = r"""
คุณมีหน้าที่จัดหมวดหมู่ Translation Memory
สำหรับระบบแปลซับไตเติลจีน -> ไทย

คุณไม่ได้แปลข้อความ
คุณไม่ได้แก้ source
คุณไม่ได้แก้ target

ให้เลือก category ได้เพียงหนึ่งค่า:

character
verb
object
terminology
expression
scene
description
other

ความหมาย:
- character = ชื่อคน ชื่อตัวละคร ชื่อเฉพาะของบุคคล
- verb = คำกริยาหรือการกระทำที่เป็นคำ/วลีสั้น
- object = สิ่งของ อาหาร สถานที่ สิ่งเฉพาะ
- terminology = ศัพท์เฉพาะ ตำแหน่ง ยศ ธุรกิจ กฎหมาย ฯลฯ
- expression = สำนวน คำพูดติดปาก วลีที่ควรจำเป็นคู่
- scene = ข้อความบรรยายฉาก เช่น [ในห้อง] หรือ [กลางคืน]
- description = ข้อความบรรยายเหตุการณ์/ภาพ เช่น [เธอเดินจากไป]
- other = อย่างอื่น

ตอบ JSON เท่านั้น:

{
  "items": [
    {
      "id": 1,
      "category": "character"
    }
  ]
}
"""


def call_ollama_json(
    system_prompt,
    user_prompt,
    timeout=MEMORY_TIMEOUT
):
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
            "num_predict": 1024,
        },
        "keep_alive": "10m",
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=timeout
    )

    response.raise_for_status()

    data = response.json()

    return (
        data.get("message", {})
        .get("content", "")
        .strip()
    )


def classify_memory_batch(items, batch_no, total_batches):
    """
    AI ทำแค่ category
    source/target ไม่ให้ AI แก้
    """

    payload_items = []

    for item in items:
        payload_items.append({
            "id": item["id"],
            "source": item["source"],
            "target": item["target"],
        })

    user_prompt = (
        "จัดหมวดหมู่รายการต่อไปนี้ "
        "โดยห้ามแก้ source และ target\n\n"
        + json.dumps(
            payload_items,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        f"   🧠 MEMORY CATEGORY "
        f"[{batch_no}/{total_batches}] "
        f"{len(items)} items"
    )

    for attempt in range(
        1,
        MEMORY_RETRIES + 1
    ):
        try:
            raw = call_ollama_json(
                MEMORY_SYSTEM_PROMPT,
                user_prompt
            )

            data = json.loads(raw)
            output = data.get("items", [])

            if not isinstance(output, list):
                raise ValueError(
                    "AI output 'items' is not a list"
                )

            result = {}

            valid_ids = {
                int(item["id"])
                for item in items
            }

            for item in output:
                try:
                    item_id = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue

                if item_id not in valid_ids:
                    continue

                # รับจาก AI เฉพาะ category เท่านั้น
                # source/target ไม่มีช่องให้ AI เปลี่ยน
                category = str(
                    item.get("category", "other")
                ).strip().lower()

                if category not in MEMORY_CATEGORIES:
                    category = "other"

                result[item_id] = category

            missing = len(items) - len(result)

            print(
                f"      ✅ classified "
                f"{len(result)}/{len(items)}"
            )

            if missing:
                print(
                    f"      ⚠️ missing category: "
                    f"{missing} "
                    f"(จะใช้ other)"
                )

            return result

        except requests.Timeout:
            print(
                f"      ⚠️ timeout "
                f"{attempt}/{MEMORY_RETRIES}"
            )

        except Exception as e:
            print(
                f"      ⚠️ category error "
                f"{attempt}/{MEMORY_RETRIES}: {e}"
            )

    print("      ❌ skip category batch")
    return {}


# ============================================================
# MEMORY UPDATE / REBUILD FROM ALL TRAINING DATA
# ============================================================

def load_all_training_examples():
    """
    อ่าน Training DB ทั้งหมด

    จุดสำคัญ:
    - ไม่ใช้เฉพาะ new_examples
    - ไม่สนว่า run ปัจจุบัน duplicate หรือไม่
    - ใช้ข้อมูลทั้งหมดที่สะสมใน translation_training_db.jsonl
    """
    examples = []

    if not os.path.exists(TRAINING_DB_JSONL):
        return examples

    with open(
        TRAINING_DB_JSONL,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = normalize_text(
                item.get("source_zh", "")
            )
            target = normalize_text(
                item.get("target_th", "")
            )

            if not source or not target:
                continue

            examples.append({
                "source": source,
                "target": target,
                "example_type": item.get(
                    "example_type",
                    "approved"
                ),
                "id": item.get("id"),
                "previous_similarity": item.get(
                    "previous_similarity",
                    0
                ),
            })

    return examples


def aggregate_training_memory_examples(examples):
    """
    รวม Source -> Target ที่เหมือนกันให้เหลือ 1 entry

    count = จำนวนครั้งที่คู่ Source -> Target
    ปรากฏใน Training DB
    """
    aggregated = {}

    for example in examples:
        source = normalize_text(
            example.get("source", "")
        )
        target = normalize_text(
            example.get("target", "")
        )

        if not source or not target:
            continue

        if not is_memory_candidate(
            source,
            target,
            example.get("example_type", "approved")
        ):
            continue

        key = memory_key(
            source,
            target
        )

        if key not in aggregated:
            aggregated[key] = {
                "source": source,
                "target": target,
                "count": 0,
                "example_types": set(),
            }

        aggregated[key]["count"] += 1
        aggregated[key]["example_types"].add(
            example.get(
                "example_type",
                "approved"
            )
        )

    # set -> list เพื่อให้ JSON serialize ได้
    for item in aggregated.values():
        item["example_types"] = sorted(
            item["example_types"]
        )

    return aggregated


def validate_memory_item(item):
    """
    ตรวจข้อมูล Memory ก่อนบันทึก
    source/target ต้องเป็นข้อมูลจาก Training DB เท่านั้น
    """
    source = normalize_text(
        item.get("source", "")
    )
    target = normalize_text(
        item.get("target", "")
    )

    if not source or not target:
        return False

    category = str(
        item.get("category", "other")
    ).strip().lower()

    if category not in MEMORY_CATEGORIES:
        category = "other"

    item["source"] = source
    item["target"] = target
    item["category"] = category

    try:
        item["count"] = max(
            1,
            int(item.get("count", 1))
        )
    except (TypeError, ValueError):
        item["count"] = 1

    return True


def save_memory(memory):
    """
    บันทึก JSONL + JSON index
    """
    os.makedirs(
        MEMORY_DIR,
        exist_ok=True
    )

    valid_items = []

    for item in memory.values():
        if validate_memory_item(item):
            valid_items.append(item)

    valid_items.sort(
        key=lambda x: (
            x.get("count", 0),
            x.get("category", ""),
            x.get("source", "")
        ),
        reverse=True
    )

    with open(
        MEMORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        for item in valid_items:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )

    by_category = {}

    for item in valid_items:
        category = item.get(
            "category",
            "other"
        )

        by_category.setdefault(
            category,
            []
        ).append(item)

    with open(
        MEMORY_INDEX_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "version": "2.0",
                "updated_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "total": len(valid_items),
                "categories": by_category,
            },
            f,
            ensure_ascii=False,
            indent=2
        )


def load_memory():
    """
    โหลด Memory เดิม

    category เดิมจะถูกเก็บไว้ เพื่อไม่ต้องเรียก AI ซ้ำ
    """
    memory = {}

    if not os.path.exists(MEMORY_FILE):
        return memory

    with open(
        MEMORY_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = normalize_text(
                item.get("source", "")
            )
            target = normalize_text(
                item.get("target", "")
            )

            if not source or not target:
                continue

            key = memory_key(
                source,
                target
            )

            item["source"] = source
            item["target"] = target

            category = str(
                item.get("category", "")
            ).strip().lower()

            if category not in MEMORY_CATEGORIES:
                category = ""

            item["category"] = category

            try:
                item["count"] = max(
                    1,
                    int(item.get("count", 1))
                )
            except (TypeError, ValueError):
                item["count"] = 1

            memory[key] = item

    return memory


def classify_uncategorized_memory(
    candidates,
    memory
):
    """
    ส่งเฉพาะรายการที่ยังไม่มี category ให้ AI

    จุดสำคัญ:
    - AI ไม่เห็นสิทธิ์ในการแก้ source/target
    - AI คืน category เท่านั้น
    - save หลังทุก batch
    - ถ้าโปรแกรมหยุดกลางทาง รอบถัดไปจะทำต่อจาก
      รายการที่ยังไม่มี category
    """

    uncategorized = []

    for item in candidates:
        key = memory_key(
            item["source"],
            item["target"]
        )

        current = memory.get(key)

        if current:
            category = str(
                current.get("category", "")
            ).strip().lower()

            if category in MEMORY_CATEGORIES:
                continue

        uncategorized.append(item)

    print(
        f"   🧠 ต้องให้ AI จัดหมวดหมู่: "
        f"{len(uncategorized)}"
    )

    if not uncategorized:
        return

    total_batches = (
        len(uncategorized) +
        MEMORY_BATCH_SIZE - 1
    ) // MEMORY_BATCH_SIZE

    for start in range(
        0,
        len(uncategorized),
        MEMORY_BATCH_SIZE
    ):
        batch = uncategorized[
            start:start + MEMORY_BATCH_SIZE
        ]

        batch_no = (
            start // MEMORY_BATCH_SIZE
        ) + 1

        classifications = classify_memory_batch(
            batch,
            batch_no,
            total_batches
        )

        # บันทึก category ที่ AI ให้มา
        for item in batch:
            item_id = item["id"]

            category = classifications.get(
                item_id,
                "other"
            )

            if category not in MEMORY_CATEGORIES:
                category = "other"

            key = memory_key(
                item["source"],
                item["target"]
            )

            if key not in memory:
                memory[key] = {
                    "source": item["source"],
                    "target": item["target"],
                    "category": category,
                    "count": item["count"],
                    "first_seen": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "last_seen": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            else:
                memory[key]["category"] = category

        # สำคัญ: save ทุก batch
        save_memory(memory)

        print(
            f"      💾 saved "
            f"{min(start + len(batch), len(uncategorized))}/"
            f"{len(uncategorized)}"
        )


def rebuild_translation_memory():
    """
    REBUILD MEMORY จาก Training DB ทั้งหมด

    Flow:

        translation_training_db.jsonl
                    ↓
             aggregate pairs
                    ↓
             existing Memory
                    ↓
       AI classify เฉพาะที่ยังไม่มี category
                    ↓
          translation_memory.jsonl

    ไม่มีการสร้าง/แก้ translation โดย AI
    """

    print("\n" + "=" * 80)
    print("TRANSLATION MEMORY")
    print("=" * 80)

    examples = load_all_training_examples()

    print(
        f"Training DB records: "
        f"{len(examples)}"
    )

    if not examples:
        print(
            "❌ ไม่พบข้อมูลใน "
            f"{TRAINING_DB_JSONL}"
        )
        return 0

    aggregated = aggregate_training_memory_examples(
        examples
    )

    print(
        f"Unique Source -> Gold pairs: "
        f"{len(aggregated)}"
    )

    memory = load_memory()

    # สร้าง/อัปเดต Memory จาก DB ทั้งหมด
    # count จะถูกคำนวณใหม่จาก Training DB เสมอ
    for key, item in aggregated.items():
        existing = memory.get(key, {})

        category = str(
            existing.get("category", "")
        ).strip().lower()

        if category not in MEMORY_CATEGORIES:
            category = ""

        memory[key] = {
            "source": item["source"],
            "target": item["target"],
            "category": category,
            "count": item["count"],
            "example_types": item[
                "example_types"
            ],
            "first_seen": existing.get(
                "first_seen",
                datetime.now().isoformat(
                    timespec="seconds"
                )
            ),
            "last_seen": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

    save_memory(memory)

    # รายการที่ยังไม่มี category
    candidates = []

    next_id = 1

    for key, item in aggregated.items():
        candidates.append({
            "id": next_id,
            "source": item["source"],
            "target": item["target"],
            "count": item["count"],
            "example_types": item[
                "example_types"
            ],
        })
        next_id += 1

    before_classification = sum(
        bool(
            str(
                memory.get(
                    memory_key(
                        item["source"],
                        item["target"]
                    ),
                    {}
                ).get("category", "")
            ).strip()
        )
        for item in candidates
    )

    print(
        f"Already categorized: "
        f"{before_classification}"
    )

    classify_uncategorized_memory(
        candidates,
        memory
    )

    # Final save
    save_memory(memory)

    categorized = 0

    for item in memory.values():
        if item.get("category") in MEMORY_CATEGORIES:
            categorized += 1

    print(
        f"\n   ✅ Total Memory: "
        f"{len(memory)}"
    )
    print(
        f"   🏷️ Categorized: "
        f"{categorized}"
    )
    print(
        f"   ⏳ Uncategorized: "
        f"{len(memory) - categorized}"
    )

    print(
        f"\n   📁 Memory:"
        f"\n   {MEMORY_FILE}"
    )
    print(
        f"\n   📁 Index:"
        f"\n   {MEMORY_INDEX_FILE}"
    )

    return len(memory)


# ============================================================
# REPORT
# ============================================================

def print_summary(
    source,
    gold,
    rag,
    results
):
    valid = [
        x for x in results
        if x["source_exists"]
        and x["gold_exists"]
        and x["rag_exists"]
    ]

    same = sum(
        x["same"]
        for x in valid
    )

    different = (
        len(valid) - same
    )

    avg = (
        sum(
            x["gold_vs_rag"]
            for x in valid
        ) / len(valid)
        if valid else 0
    )

    print("\n" + "=" * 80)
    print("SRT RAG COMPARISON")
    print("=" * 80)

    print(f"\nSOURCE:\n{source}")
    print(f"\nGOLD STANDARD:\n{gold}")
    print(f"\nRAG OUTPUT:\n{rag}")

    print("\n" + "-" * 80)
    print(
        f"จำนวน subtitle ทั้งหมด: "
        f"{len(results)}"
    )
    print(
        f"Gold และ RAG เหมือนกัน: "
        f"{same}"
    )
    print(
        f"Gold และ RAG ต่างกัน: "
        f"{different}"
    )
    print(
        f"ค่าเฉลี่ยความเหมือน RAG <-> Gold: "
        f"{avg:.2%}"
    )


def print_differences(results):
    differences = [
        x for x in results
        if x["gold_exists"]
        and x["rag_exists"]
        and not x["same"]
    ]

    print("\n" + "=" * 80)
    print(
        f"DIFFERENCES: "
        f"{len(differences)}"
    )
    print("=" * 80)

    for item in differences:
        print("\n" + "-" * 80)
        print(
            f"[{item['index']}] "
            f"{item['timecode']}"
        )

        print("\nSOURCE:")
        print(item["source"])

        print("\nGOLD:")
        print(item["gold"])

        print("\nRAG:")
        print(item["rag"])

        print(
            f"\nGold <-> RAG: "
            f"{item['gold_vs_rag']:.2%}"
        )


def save_report(
    path,
    source,
    gold,
    rag,
    results
):
    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "SRT RAG COMPARISON REPORT\n"
        )
        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"SOURCE:\n{source}\n\n"
        )
        f.write(
            f"GOLD STANDARD:\n{gold}\n\n"
        )
        f.write(
            f"RAG OUTPUT:\n{rag}\n\n"
        )

        valid = [
            x for x in results
            if x["source_exists"]
            and x["gold_exists"]
            and x["rag_exists"]
        ]

        same = sum(
            x["same"]
            for x in valid
        )

        different = (
            len(valid) - same
        )

        avg = (
            sum(
                x["gold_vs_rag"]
                for x in valid
            ) / len(valid)
            if valid else 0
        )

        f.write(
            f"TOTAL: {len(results)}\n"
        )
        f.write(
            f"SAME: {same}\n"
        )
        f.write(
            f"DIFFERENT: {different}\n"
        )
        f.write(
            f"AVERAGE GOLD/RAG SIMILARITY: "
            f"{avg:.2%}\n\n"
        )

        f.write(
            "=" * 80 + "\n"
        )
        f.write(
            "DIFFERENCES\n"
        )
        f.write(
            "=" * 80 + "\n"
        )

        for item in results:
            if not item["gold_exists"]:
                continue

            if not item["rag_exists"]:
                continue

            if item["same"]:
                continue

            f.write(
                f"\n[{item['index']}] "
                f"{item['timecode']}\n"
            )
            f.write(
                f"\nSOURCE:\n"
                f"{item['source']}\n"
            )
            f.write(
                f"\nGOLD:\n"
                f"{item['gold']}\n"
            )
            f.write(
                f"\nRAG:\n"
                f"{item['rag']}\n"
            )
            f.write(
                f"\nSimilarity: "
                f"{item['gold_vs_rag']:.2%}\n"
            )
            f.write(
                "-" * 80 + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(
        INPUT_DIR,
        exist_ok=True
    )
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )
    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )
    os.makedirs(
        MEMORY_DIR,
        exist_ok=True
    )

    source_file = select_file(
        INPUT_DIR,
        "1. เลือกไฟล์ต้นฉบับภาษาจีน"
    )

    if not source_file:
        return

    gold_file = select_file(
        OUTPUT_DIR,
        "2. เลือกไฟล์ Gold Standard"
    )

    if not gold_file:
        return

    rag_file = select_file(
        OUTPUT_DIR,
        "3. เลือกไฟล์ RAG Output"
    )

    if not rag_file:
        return

    print("\n" + "=" * 80)
    print("SELECTED FILES")
    print("=" * 80)

    print(f"\nSOURCE:\n{source_file}")
    print(f"\nGOLD:\n{gold_file}")
    print(f"\nRAG:\n{rag_file}")

    print("\nกำลังเปรียบเทียบ...")

    results = compare_srt(
        source_file,
        gold_file,
        rag_file
    )

    print_summary(
        source_file,
        gold_file,
        rag_file,
        results
    )

    print_differences(results)

    # --------------------------------------------------------
    # Comparison report
    # --------------------------------------------------------

    report_path = os.path.join(
        REPORT_DIR,
        "comparison.txt"
    )

    save_report(
        report_path,
        source_file,
        gold_file,
        rag_file,
        results
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    (
        new_examples,
        duplicate_in_run,
        duplicate_from_history
    ) = export_training_data(
        results,
        source_file,
        gold_file,
        rag_file
    )

    print("\n" + "=" * 80)
    print("TRAINING EXPORT")
    print("=" * 80)

    print(
        f"\nNew training examples: "
        f"{len(new_examples)}"
    )

    print(
        f"Duplicates in current run: "
        f"{duplicate_in_run}"
    )

    print(
        f"Duplicates from history: "
        f"{duplicate_from_history}"
    )

    print(
        f"\n1. {TRAINING_JSONL}"
    )
    print(
        f"2. {TRAINING_JSON}"
    )
    print(
        f"3. {FEEDBACK_JSON}"
    )
    print(
        f"4. {TRAINING_DB_JSONL}"
    )

    # --------------------------------------------------------
    # Translation Memory
    # --------------------------------------------------------
    #
    # สำคัญ:
    # Memory ไม่ใช้เฉพาะ new_examples
    # แต่ rebuild จาก translation_training_db.jsonl ทั้งหมด
    #
    rebuild_translation_memory()

    print("\n" + "=" * 80)
    print("FILES")
    print("=" * 80)

    print(
        f"\nComparison report:\n"
        f"{report_path}"
    )

    print(
        f"\nTranslation Memory:\n"
        f"{MEMORY_FILE}"
    )

    print(
        f"\nMemory index:\n"
        f"{MEMORY_INDEX_FILE}"
    )

    print("\n" + "=" * 80)
    print("เสร็จสิ้น")
    print("=" * 80)


if __name__ == "__main__":
    main()
