# -*- coding: utf-8 -*-
"""
import_srt_memory.py

นำ SRT ของคนแปลเข้า Translation Memory โดยตรง

ต้องมี 2 ไฟล์:
1. source.srt       = SRT ภาษาจีนต้นฉบับ
2. translated.srt  = SRT ภาษาไทยของคนแปล

Flow:
SRT จีน + SRT คนแปล
        ↓
จับคู่ subtitle index
        ↓
ส่งคู่ Source/Gold ให้ AI Extract คำ/วลี
        ↓
AI จัด category
        ↓
ตรวจ duplicate / conflict
        ↓
comparison/memory/translation_memory.jsonl
comparison/memory/translation_memory.json

AI มีหน้าที่ Extract + จัดหมวดเท่านั้น
ไม่ได้สร้างคำแปลใหม่

ใช้ Ollama ผ่าน HTTP API
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MEMORY_DIR = BASE_DIR / "comparison" / "memory"

MEMORY_JSONL = MEMORY_DIR / "translation_memory.jsonl"
MEMORY_JSON = MEMORY_DIR / "translation_memory.json"

OLLAMA_URL = "http://localhost:11434/api/chat"

# เปลี่ยนได้ตาม Ollama ที่คุณใช้
OLLAMA_MODEL = "qwen3:4b"

AI_TIMEOUT = 300
AI_RETRIES = 3

# จำนวน subtitle ต่อ batch
BATCH_SIZE = 20

# ============================================================
# CATEGORY
# ============================================================

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
# TEXT UTILS
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")
    text = text.strip()

    # whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def clean_json_response(text):
    """
    เอา JSON จาก response ของ AI
    รองรับ ```json ... ```
    """
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    return text.strip()


# ============================================================
# SRT PARSER
# ============================================================

def parse_srt(path):
    """
    คืน:
    {
        index: {
            "index": int,
            "timecode": "...",
            "text": "..."
        }
    }
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์: {path}"
        )

    raw = None

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "utf-8-sig",
        "cp874",
        "cp932",
        "gb18030",
    ):
        try:
            raw = path.read_text(
                encoding=encoding
            )
            break
        except UnicodeDecodeError:
            continue

    if raw is None:
        raise UnicodeDecodeError(
            "unknown",
            b"",
            0,
            1,
            f"อ่านไฟล์ไม่ได้: {path}"
        )

    raw = raw.replace("\r\n", "\n")
    raw = raw.replace("\r", "\n")

    blocks = re.split(
        r"\n\s*\n",
        raw.strip()
    )

    result = {}

    for block in blocks:

        lines = block.split("\n")

        if len(lines) < 3:
            continue

        try:
            index = int(
                lines[0].strip()
            )
        except ValueError:
            continue

        timecode = lines[1].strip()

        text = "\n".join(
            lines[2:]
        ).strip()

        text = text.replace(
            r"\N",
            "\n"
        )

        if not text:
            continue

        result[index] = {
            "index": index,
            "timecode": timecode,
            "text": text,
        }

    return result


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
        },
    }

    for attempt in range(
        1,
        AI_RETRIES + 1
    ):

        try:

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=AI_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            result = data.get(
                "response",
                ""
            )

            if result:
                return result.strip()

            if not result:
                print("⚠️ Ollama ไม่มี response")
                print("Ollama raw:", data)
                return ""
            
            return result.strip()
        
            print(
                f"⚠️ Ollama ไม่มี response "
                f"(attempt {attempt})"
            )

        except Exception as e:

            print(
                f"⚠️ Ollama error "
                f"(attempt {attempt}): {e}"
            )

            if attempt < AI_RETRIES:
                time.sleep(2)

    return ""


# ============================================================
# AI EXTRACTION
# ============================================================

def build_extract_prompt(items):
    """
    AI ได้ Source + Gold ของคนแปล

    ต้อง Extract เฉพาะคำ/วลีที่:
    - มีประโยชน์ในการแปลซ้ำ
    - เป็นชื่อคน
    - สถานที่
    - คำกริยา
    - สิ่งของ
    - ศัพท์เฉพาะ
    - สำนวน
    - ฉาก
    - คำอธิบาย
    - หรือคำ/วลีสำคัญอื่น

    ห้ามคิดคำแปลใหม่
    target ต้องมาจาก Gold จริง
    """

    lines = []

    for item in items:

        lines.append(
            f'ID: {item["index"]}\n'
            f'SOURCE: {item["source"]}\n'
            f'GOLD: {item["target"]}'
        )

    joined = "\n\n".join(lines)

    return f"""
คุณเป็นระบบสร้าง Translation Memory สำหรับการแปล
ภาษาจีน → ภาษาไทย

หน้าที่ของคุณคือ "Extract คำ/วลีจากคำแปลคนแปล"
และจัดหมวดหมู่เท่านั้น

ห้ามสร้างคำแปลใหม่
ห้ามแก้คำแปลของ Gold
ห้ามเดาคำแปลจากความรู้ภายนอก

ให้ "แบ่งและ Extract คำ/วลีสำคัญ" จาก SOURCE → GOLD

หมายถึง:
- แยกคำสำคัญออกจากประโยค
- ถ้าเป็นคำเดี่ยวที่มีประโยชน์ ให้เก็บเป็นคำเดี่ยว
- ถ้าเป็นชื่อเฉพาะหรือวลีที่ต้องจำเป็นชุด ให้เก็บเป็นวลี
- ไม่ต้องแบ่งทุกคำในประโยค
- ห้ามเก็บประโยคเต็ม เว้นแต่เป็นสำนวนหรือ expression ที่ควรจำทั้งชุด

โดยเฉพาะ:
- ชื่อคน
- สถานที่
- คำกริยา
- สิ่งของ
- ศัพท์เฉพาะ
- สำนวน
- ฉาก
- คำอธิบาย
- คำ/วลีสำคัญอื่น

CATEGORY ที่อนุญาตเท่านั้น:
character
place
verb
object
terminology
expression
scene
description
other

กฎสำคัญ:
1. source ต้องเป็นคำหรือวลีที่มีอยู่จริงใน SOURCE
2. target ต้องเป็นคำหรือวลีที่มีอยู่จริงใน GOLD
3. ห้ามสร้าง target ใหม่
4. ห้ามเปลี่ยนความหมายของ Gold
5. ถ้าไม่แน่ใจ ไม่ต้อง Extract
6. ไม่จำเป็นต้อง Extract ทุกคำ
7. คำทั่วไปที่ไม่ช่วยให้การแปลครั้งหน้าสม่ำเสมอ ไม่ต้องเก็บ
8. ถ้า SOURCE เป็นชื่อคน/สถานที่ ต้องให้ความสำคัญ
9. Extract ได้หลายรายการจาก subtitle เดียว
10. ห้ามส่ง subtitle ทั้งประโยคเป็น Memory เว้นแต่เป็นสำนวนหรือ expression ที่ควรจำทั้งประโยค

ส่งกลับเป็น JSON ARRAY เท่านั้น:

[
  {{
    "id": 123,
    "source": "周乾",
    "target": "โจวเฉียน",
    "category": "character"
  }}
]

ข้อมูล:

{joined}
""".strip()


def validate_candidate(
    candidate,
    source,
    target,
):
    """
    ตรวจว่า AI ไม่ได้สร้างข้อมูลมั่ว
    """

    if not isinstance(
        candidate,
        dict
    ):
        return None

    candidate_source = normalize_text(
        candidate.get("source", "")
    )

    candidate_target = normalize_text(
        candidate.get("target", "")
    )

    category = normalize_text(
        candidate.get("category", "")
    ).lower()

    if not candidate_source:
        return None

    if not candidate_target:
        return None

    if category not in MEMORY_CATEGORIES:
        category = "other"

    # source ต้องอยู่ใน SOURCE
    if candidate_source not in source:
        return None

    # target ต้องอยู่ใน Gold
    if candidate_target not in target:
        return None

    return {
        "source": candidate_source,
        "target": candidate_target,
        "category": category,
    }


def extract_batch(items):
    if not items:
        return []

    prompt = build_extract_prompt(
        items
    )

    response = call_ollama(
        prompt
    )

    if not response:
        return []

    response = clean_json_response(
        response
    )

    try:
        data = json.loads(
            response
        )
    except json.JSONDecodeError:

        # พยายามดึง array ออก
        match = re.search(
            r"\[[\s\S]*\]",
            response
        )

        if not match:
            print(
                "⚠️ AI response ไม่ใช่ JSON"
            )
            return []

        try:
            data = json.loads(
                match.group(0)
            )
        except json.JSONDecodeError:
            print(
                "⚠️ อ่าน JSON จาก AI ไม่ได้"
            )
            return []

    if not isinstance(
        data,
        list
    ):
        return []

    item_map = {
        item["index"]: item
        for item in items
    }

    results = []

    for candidate in data:

        if not isinstance(
            candidate,
            dict
        ):
            continue

        try:
            item_id = int(
                candidate.get(
                    "id"
                )
            )
        except (
            TypeError,
            ValueError
        ):
            continue

        item = item_map.get(
            item_id
        )

        if not item:
            continue

        valid = validate_candidate(
            candidate,
            item["source"],
            item["target"],
        )

        if valid:
            valid["subtitle_id"] = (
                item_id
            )
            results.append(
                valid
            )

    return results


# ============================================================
# MEMORY LOAD
# ============================================================

def load_memory():
    """
    โหลด Memory เดิม

    รองรับ schema:
    source / target
    source_zh / target_th
    """

    memory = []

    if not MEMORY_JSONL.exists():
        return memory

    with open(
        MEMORY_JSONL,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(
                    line
                )
            except json.JSONDecodeError:
                continue

            source = normalize_text(
                item.get("source")
                or item.get("source_zh")
                or ""
            )

            target = normalize_text(
                item.get("target")
                or item.get("target_th")
                or ""
            )

            if not source or not target:
                continue

            category = normalize_text(
                item.get(
                    "category",
                    "other"
                )
            ).lower()

            if category not in MEMORY_CATEGORIES:
                category = "other"

            try:
                count = int(
                    item.get(
                        "count",
                        1
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                count = 1

            memory.append({
                "source": source,
                "target": target,
                "category": category,
                "count": count,
            })

    return memory


def memory_key(
    source,
    target
):
    return (
        normalize_text(source),
        normalize_text(target),
    )


# ============================================================
# MEMORY MERGE
# ============================================================

def merge_candidates(
    memory,
    candidates
):
    """
    Merge candidate เข้ากับ Memory

    เหมือน memory_input.xlsx:

    1. source + target เหมือน
       → count++

    2. source เหมือน แต่ target ต่าง
       → Conflict
       → ถามผู้ใช้

    3. source ใหม่
       → เพิ่มทันที
    """

    # index:
    # source -> list ของ memory entries
    by_source = defaultdict(list)

    for item in memory:
        by_source[
            normalize_text(
                item["source"]
            )
        ].append(item)

    added = 0
    updated = 0
    conflicts = 0

    for candidate in candidates:

        source = candidate["source"]
        target = candidate["target"]
        category = candidate["category"]

        source_entries = by_source.get(
            source,
            []
        )

        # ----------------------------------------------------
        # Source ไม่มี
        # ----------------------------------------------------

        if not source_entries:

            new_item = {
                "source": source,
                "target": target,
                "category": category,
                "count": 1,
            }

            memory.append(
                new_item
            )

            by_source[
                source
            ].append(
                new_item
            )

            added += 1

            print(
                f"   ➕ NEW "
                f"{source} → {target} "
                f"[{category}]"
            )

            continue

        # ----------------------------------------------------
        # Source + Target เหมือน
        # ----------------------------------------------------

        exact = None

        for existing in source_entries:

            if normalize_text(
                existing["target"]
            ) == target:

                exact = existing
                break

        if exact:

            exact["count"] = (
                int(
                    exact.get(
                        "count",
                        1
                    )
                ) + 1
            )

            # ถ้าของเดิมไม่มี category
            if not exact.get(
                "category"
            ):
                exact[
                    "category"
                ] = category

            updated += 1

            print(
                f"   🔄 EXIST "
                f"{source} → {target} "
                f"(count={exact['count']})"
            )

            continue

        # ----------------------------------------------------
        # Source เหมือน แต่ Target ต่าง
        # ----------------------------------------------------

        conflicts += 1

        print(
            "\n" + "=" * 70
        )
        print(
            "⚠️ MEMORY CONFLICT"
        )
        print(
            f"SOURCE: {source}"
        )

        print(
            "Memory เดิม:"
        )

        for i, existing in enumerate(
            source_entries,
            1
        ):
            print(
                f"  [{i}] "
                f"{existing['target']} "
                f"(count={existing.get('count', 1)}) "
                f"[{existing.get('category', 'other')}]"
            )

        print(
            f"\nข้อมูลใหม่:"
        )
        print(
            f"  {target} "
            f"[{category}]"
        )

        print(
            "\nเลือก:"
        )
        print(
            "[1] ใช้ Memory เดิม "
            "(ข้ามข้อมูลใหม่)"
        )
        print(
            "[2] เพิ่มคำแปลใหม่ "
            "(เก็บทั้งสอง)"
        )
        print(
            "[3] เปลี่ยน Memory เดิม "
            "(ลบของเดิมและใช้ใหม่)"
        )
        print(
            "[4] ข้าม"
        )

        while True:

            choice = input(
                "เลือก 1/2/3/4: "
            ).strip()

            if choice in {
                "1",
                "2",
                "3",
                "4",
            }:
                break

        if choice == "1":

            print(
                "   ↪ ใช้ Memory เดิม"
            )

            continue

        if choice == "2":

            new_item = {
                "source": source,
                "target": target,
                "category": category,
                "count": 1,
            }

            memory.append(
                new_item
            )

            by_source[
                source
            ].append(
                new_item
            )

            added += 1

            print(
                "   ➕ เก็บทั้งสองคำแปล"
            )

        elif choice == "3":

            # ลบ entries ของ source นี้
            old_entries = list(
                by_source[source]
            )

            for old in old_entries:

                if old in memory:
                    memory.remove(
                        old
                    )

            new_item = {
                "source": source,
                "target": target,
                "category": category,
                "count": 1,
            }

            memory.append(
                new_item
            )

            by_source[
                source
            ] = [new_item]

            updated += 1

            print(
                "   🔄 เปลี่ยนเป็นคำแปลใหม่"
            )

        elif choice == "4":

            print(
                "   ⏭ ข้าม"
            )

    return (
        memory,
        added,
        updated,
        conflicts,
    )


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(memory):
    MEMORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # sort:
    # category → source
    memory = sorted(
        memory,
        key=lambda x: (
            x.get(
                "category",
                "other"
            ),
            x["source"],
            x["target"],
        )
    )

    with open(
        MEMORY_JSONL,
        "w",
        encoding="utf-8"
    ) as f:

        for item in memory:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    # index JSON
    index = {
        "total": len(memory),
        "categories": {},
        "items": memory,
    }

    category_count = defaultdict(
        int
    )

    for item in memory:
        category_count[
            item.get(
                "category",
                "other"
            )
        ] += 1

    index[
        "categories"
    ] = dict(
        sorted(
            category_count.items()
        )
    )

    with open(
        MEMORY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            index,
            f,
            ensure_ascii=False,
            indent=2
        )

    return memory



# ============================================================
# INTERACTIVE FILE SELECTION
# ============================================================

def clean_path_input(value):
    return value.strip().strip('"').strip("'")


def list_srt_files(folder):
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [x for x in folder.iterdir()
         if x.is_file() and x.suffix.lower() == ".srt"],
        key=lambda x: x.name.lower()
    )


def choose_srt_file(title, folder=None):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)

    if folder:
        files = list_srt_files(folder)
        if files:
            print(f"พบ SRT {len(files)} ไฟล์ใน:")
            print(f"  {folder}\n")
            for i, file in enumerate(files, 1):
                print(f"[{i}] {file.name}")
            print("[0] ใส่ path เอง")

            while True:
                choice = input("เลือกไฟล์: ").strip()
                if choice == "0":
                    break
                try:
                    n = int(choice)
                    if 1 <= n <= len(files):
                        return str(files[n - 1])
                except ValueError:
                    pass
                print("❌ เลือกหมายเลขไม่ถูกต้อง")

    while True:
        value = clean_path_input(input("ใส่ path ของไฟล์ .srt: "))
        if value and Path(value).exists() and Path(value).suffix.lower() == ".srt":
            return value
        print("❌ ไม่พบไฟล์ .srt กรุณาลองใหม่")


def choose_folder(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    while True:
        folder = clean_path_input(input("ใส่ path โฟลเดอร์: "))
        if folder and Path(folder).is_dir():
            return Path(folder)
        print("❌ ไม่พบโฟลเดอร์ กรุณาลองใหม่")


def select_input_files():
    print("\n" + "=" * 80)
    print("SELECT SRT")
    print("=" * 80)
    print("[1] ใส่ path Source และ Gold เอง")
    print("[2] เลือก Source / Gold จากคนละโฟลเดอร์")
    print("[3] เลือก Source / Gold จากโฟลเดอร์เดียวกัน")

    while True:
        choice = input("\nเลือก 1/2/3: ").strip()
        if choice in {"1", "2", "3"}:
            break
        print("❌ กรุณาเลือก 1, 2 หรือ 3")

    if choice == "1":
        return (
            choose_srt_file("SOURCE SRT (จีนต้นฉบับ)"),
            choose_srt_file("GOLD SRT (คนแปล)")
        )

    if choice == "2":
        source_folder = choose_folder("โฟลเดอร์ SOURCE SRT")
        target_folder = choose_folder("โฟลเดอร์ GOLD SRT")
        return (
            choose_srt_file("SOURCE SRT (จีนต้นฉบับ)", source_folder),
            choose_srt_file("GOLD SRT (คนแปล)", target_folder)
        )

    folder = choose_folder("โฟลเดอร์ที่มีทั้ง SOURCE และ GOLD SRT")
    return (
        choose_srt_file("SOURCE SRT (จีนต้นฉบับ)", folder),
        choose_srt_file("GOLD SRT (คนแปล)", folder)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )
    print(
        "SRT → TRANSLATION MEMORY"
    )
    print(
        "=" * 80
    )

    print(
        "\nต้องใช้:"
    )
    print(
        "1. source.srt       = จีนต้นฉบับ"
    )
    print(
        "2. translated.srt  = SRT คนแปล"
    )

    source_path, target_path = select_input_files()

    print(f"\nSOURCE:\n  {source_path}")
    print(f"\nGOLD:\n  {target_path}")

    try:
        source_data = parse_srt(
            source_path
        )

        target_data = parse_srt(
            target_path
        )

    except Exception as e:

        print(
            f"❌ อ่าน SRT ไม่สำเร็จ: {e}"
        )
        return

    print(
        f"\nSource subtitles: "
        f"{len(source_data)}"
    )

    print(
        f"Gold subtitles: "
        f"{len(target_data)}"
    )

    # --------------------------------------------------------
    # Match subtitles
    # --------------------------------------------------------

    items = []

    for index in sorted(
        set(source_data)
        & set(target_data)
    ):

        source = normalize_text(
            source_data[index]["text"]
        )

        target = normalize_text(
            target_data[index]["text"]
        )

        if not source or not target:
            continue

        items.append({
            "index": index,
            "source": source,
            "target": target,
            "timecode": source_data[
                index
            ]["timecode"],
        })

    print(
        f"Matched subtitles: "
        f"{len(items)}"
    )

    print("\nโหมด Memory: แบ่งเป็นคำ/วลีสำคัญ")
    print("AI จะไม่เก็บทุกคำ และจะไม่สร้างคำแปลใหม่")

    if not items:
        print(
            "❌ ไม่พบ subtitle ที่จับคู่กันได้"
        )
        return

    # --------------------------------------------------------
    # Load existing Memory
    # --------------------------------------------------------

    memory = load_memory()

    print(
        f"\nExisting Memory: "
        f"{len(memory)}"
    )

    # --------------------------------------------------------
    # AI extraction
    # --------------------------------------------------------

    print(
        "\n🧠 กำลังให้ AI "
        "Extract คำ/วลีจาก Gold..."
    )

    all_candidates = []

    total_batches = (
        len(items)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for batch_no, start in enumerate(
        range(
            0,
            len(items),
            BATCH_SIZE
        ),
        1
    ):

        batch = items[
            start:
            start + BATCH_SIZE
        ]

        print(
            f"\n🔄 Batch "
            f"{batch_no}/{total_batches} "
            f"({len(batch)} subtitles)"
        )

        candidates = extract_batch(
            batch
        )

        print(
            f"   ได้ Candidate: "
            f"{len(candidates)}"
        )

        all_candidates.extend(
            candidates
        )

        time.sleep(0.2)

    # --------------------------------------------------------
    # Deduplicate candidates
    # --------------------------------------------------------

    unique = {}

    for candidate in all_candidates:

        key = (
            candidate["source"],
            candidate["target"],
            candidate["category"],
        )

        unique[key] = candidate

    candidates = list(
        unique.values()
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "EXTRACTION RESULT"
    )

    print(
        "=" * 80
    )

    print(
        f"AI candidates: "
        f"{len(all_candidates)}"
    )

    print(
        f"Unique candidates: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Show candidates before merge
    # --------------------------------------------------------

    for candidate in candidates:

        print(
            f"  "
            f"{candidate['source']} "
            f"→ "
            f"{candidate['target']} "
            f"[{candidate['category']}]"
        )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "MERGE TO TRANSLATION MEMORY"
    )

    print(
        "=" * 80
    )

    (
        memory,
        added,
        updated,
        conflicts,
    ) = merge_candidates(
        memory,
        candidates
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_memory(
        memory
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "เสร็จสิ้น"
    )

    print(
        "=" * 80
    )

    print(
        f"➕ เพิ่มใหม่: {added}"
    )

    print(
        f"🔄 Update/count: {updated}"
    )

    print(
        f"⚠️ Conflict: {conflicts}"
    )

    print(
        f"🧠 Memory ทั้งหมด: "
        f"{len(memory)}"
    )

    print(
        "\nMemory:"
    )

    print(
        MEMORY_JSONL
    )

    print(
        "\nIndex:"
    )

    print(
        MEMORY_JSON
    )


if __name__ == "__main__":
    main()
