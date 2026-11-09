import os
import json
import glob
import requests
from openpyxl import load_workbook

# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = "memory_input"

MEMORY_DIR = os.path.join(
    "comparison",
    "memory"
)

MEMORY_FILE = os.path.join(
    MEMORY_DIR,
    "translation_memory.jsonl"
)

MEMORY_INDEX_FILE = os.path.join(
    MEMORY_DIR,
    "translation_memory.json"
)

OLLAMA_URL = "http://localhost:11434/api/chat"

# เปลี่ยนตรงนี้ได้ตาม model ที่ใช้
MODEL = "qwen3:4b"

# ============================================================
# CATEGORY
# ============================================================

VALID_CATEGORIES = {
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
# NORMALIZE
# ============================================================

def normalize(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    return " ".join(text.split()).strip()


def normalize_key(text):
    return normalize(text).lower()


# ============================================================
# LOAD MEMORY
# ============================================================

def load_memory():

    memory = []

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

                if isinstance(item, dict):
                    memory.append(item)

            except Exception:
                print(
                    "⚠️ อ่าน Memory บรรทัดไม่ได้:"
                )
                print(line)

    return memory


# ============================================================
# SAVE MEMORY
# ============================================================

def save_memory(memory):

    os.makedirs(
        MEMORY_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    with open(
        MEMORY_FILE,
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

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    categories = {}

    for item in memory:

        category = item.get(
            "category",
            "other"
        )

        categories.setdefault(
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
                "total": len(memory),
                "categories": categories,
            },
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# READ EXCEL
# ============================================================

def read_excel(path):

    workbook = load_workbook(
        path,
        read_only=True,
        data_only=True
    )

    sheets = []

    for ws in workbook.worksheets:

        rows = []

        for row in ws.iter_rows(
            values_only=True
        ):

            values = [
                normalize(value)
                for value in row
            ]

            # ข้ามแถวว่าง
            if not any(values):
                continue

            rows.append(values)

        if rows:

            sheets.append(
                {
                    "sheet": ws.title,
                    "rows": rows
                }
            )

    workbook.close()

    return sheets


# ============================================================
# CONVERT EXCEL TO TEXT
# ============================================================

def excel_to_text(excel_data):

    output = []

    for sheet in excel_data:

        output.append(
            f"=== SHEET: {sheet['sheet']} ==="
        )

        for row_number, row in enumerate(
            sheet["rows"],
            start=1
        ):

            values = []

            for col_number, value in enumerate(
                row,
                start=1
            ):

                if not value:
                    continue

                values.append(
                    f"C{col_number}={value}"
                )

            if values:

                output.append(
                    f"ROW {row_number}: "
                    + " | ".join(values)
                )

    return "\n".join(output)


# ============================================================
# OLLAMA
# ============================================================

def ask_ollama(excel_text):

    system_prompt = """
คุณเป็นระบบจัดข้อมูล Translation Memory
สำหรับแปล subtitle ภาษาจีน → ภาษาไทย

หน้าที่ของคุณคือ READ และ CLASSIFY ข้อมูลจาก Excel เท่านั้น

สำคัญมาก:

1. ห้ามแก้ source
2. ห้ามแก้ target
3. ห้ามแปลข้อความใหม่
4. ห้ามเดาคำแปล
5. ใช้คำแปลที่มีอยู่ใน Excel เท่านั้น
6. ถ้าไม่แน่ใจ ให้ category = other
7. ต้องคืนเฉพาะข้อมูลที่มี source และ target จริง

หมวดหมู่ที่อนุญาต:

character
place
verb
object
terminology
expression
scene
description
other

ความหมาย:

character = ชื่อคน/ตัวละคร
place = ชื่อสถานที่
verb = คำกริยา
object = สิ่งของ
terminology = ศัพท์เฉพาะ/ชื่อองค์กร/แบรนด์
expression = สำนวน/วลี
scene = ข้อความฉาก เช่น [เสียงฝน]
description = คำบรรยาย
other = อื่นๆ

ต้องตอบ JSON array เท่านั้น:

[
  {
    "source": "...",
    "target": "...",
    "category": "...",
    "note": "..."
  }
]

ถ้าไม่มีข้อมูลที่ใช้ได้ ให้ตอบ:

[]
"""

    user_prompt = f"""
อ่านข้อความจาก Excel ด้านล่าง

{excel_text}

ดึงเฉพาะคู่:
SOURCE → TARGET

และจัด category ให้แต่ละรายการ

จำไว้ว่า:
ห้ามแปลใหม่
ห้ามแก้คำแปล
ห้ามสร้างข้อมูลที่ไม่มีใน Excel
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        data = response.json()

        content = data.get(
            "message",
            {}
        ).get(
            "content",
            ""
        )

        return parse_ai_json(content)

    except Exception as e:

        print(
            f"\n❌ Ollama error: {e}"
        )

        return []


# ============================================================
# PARSE AI JSON
# ============================================================

def parse_ai_json(text):

    text = text.strip()

    # --------------------------------------------------------
    # เอา markdown fence ออก
    # --------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        lines = [
            line
            for line in lines
            if not line.strip().startswith("```")
        ]

        text = "\n".join(lines)

    # --------------------------------------------------------
    # หา JSON array
    # --------------------------------------------------------

    start = text.find("[")

    end = text.rfind("]")

    if start == -1 or end == -1:

        print(
            "\n⚠️ AI ไม่ได้ส่ง JSON array"
        )

        print(text[:1000])

        return []

    json_text = text[
        start:end + 1
    ]

    try:

        data = json.loads(
            json_text
        )

    except Exception as e:

        print(
            f"\n⚠️ JSON parse error: {e}"
        )

        print(
            json_text[:1000]
        )

        return []

    if not isinstance(
        data,
        list
    ):

        return []

    return data


# ============================================================
# VALIDATE AI RESULT
# ============================================================

def validate_item(item):

    if not isinstance(
        item,
        dict
    ):
        return None

    source = normalize(
        item.get(
            "source",
            ""
        )
    )

    target = normalize(
        item.get(
            "target",
            ""
        )
    )

    category = normalize(
        item.get(
            "category",
            "other"
        )
    ).lower()

    note = normalize(
        item.get(
            "note",
            ""
        )
    )

    if not source:
        return None

    if not target:
        return None

    if category not in VALID_CATEGORIES:

        category = "other"

    result = {
        "source": source,
        "target": target,
        "category": category,
        "count": 1,
        "source_type": "excel_ai"
    }

    if note:
        result["note"] = note

    return result


# ============================================================
# FIND EXISTING
# ============================================================

def find_existing(
    memory,
    source
):

    key = normalize_key(
        source
    )

    return [
        item
        for item in memory
        if normalize_key(
            item.get(
                "source",
                ""
            )
        ) == key
    ]


# ============================================================
# CONFLICT
# ============================================================

def ask_conflict(
    old_item,
    new_item
):

    print("\n")
    print("=" * 80)
    print("⚠️ SOURCE ซ้ำ แต่คำแปลไม่เหมือนกัน")
    print("=" * 80)

    print(
        f"\nSOURCE:\n"
        f"{new_item['source']}"
    )

    print(
        f"\nMemory เดิม:\n"
        f"{old_item.get('target', '')}"
    )

    print(
        f"\nExcel ใหม่:\n"
        f"{new_item['target']}"
    )

    print(
        f"\nCategory เดิม:"
        f" {old_item.get('category', 'other')}"
    )

    print(
        f"Category ใหม่:"
        f" {new_item.get('category', 'other')}"
    )

    print(
        "\nเลือก:"
    )

    print(
        "[1] ใช้ Memory เดิม"
    )

    print(
        "[2] ใช้คำแปลใหม่"
    )

    print(
        "[3] เก็บทั้งสอง"
    )

    print(
        "[4] ข้าม"
    )

    while True:

        choice = input(
            "\nเลือก 1/2/3/4: "
        ).strip()

        if choice in {
            "1",
            "2",
            "3",
            "4"
        }:

            return choice

        print(
            "กรุณาเลือก 1-4"
        )


# ============================================================
# MERGE
# ============================================================

def merge_item(
    memory,
    new_item
):

    existing = find_existing(
        memory,
        new_item["source"]
    )

    # --------------------------------------------------------
    # NEW
    # --------------------------------------------------------

    if not existing:

        memory.append(
            new_item
        )

        print(
            f"🟢 NEW  "
            f"{new_item['source']} "
            f"→ "
            f"{new_item['target']}"
        )

        return "new"

    # --------------------------------------------------------
    # SAME TARGET
    # --------------------------------------------------------

    for old_item in existing:

        if normalize_key(
            old_item.get(
                "target",
                ""
            )
        ) == normalize_key(
            new_item["target"]
        ):

            old_count = old_item.get(
                "count",
                1
            )

            try:
                old_count = int(
                    old_count
                )
            except Exception:
                old_count = 1

            old_item["count"] = (
                old_count + 1
            )

            # รวม note
            new_note = new_item.get(
                "note",
                ""
            )

            old_note = old_item.get(
                "note",
                ""
            )

            if new_note:

                if not old_note:

                    old_item["note"] = new_note

                elif new_note not in old_note:

                    old_item["note"] = (
                        old_note
                        + " | "
                        + new_note
                    )

            print(
                f"⚪ DUPLICATE "
                f"{new_item['source']} "
                f"→ "
                f"{new_item['target']}"
            )

            return "duplicate"

    # --------------------------------------------------------
    # CONFLICT
    # --------------------------------------------------------

    choice = ask_conflict(
        existing[0],
        new_item
    )

    if choice == "1":

        print(
            "→ ใช้ Memory เดิม"
        )

        return "keep_old"

    if choice == "2":

        old_item = existing[0]

        old_item["target"] = (
            new_item["target"]
        )

        old_item["category"] = (
            new_item["category"]
        )

        if new_item.get("note"):

            old_item["note"] = (
                new_item["note"]
            )

        old_item["source_type"] = (
            "excel_ai"
        )

        print(
            "→ เปลี่ยนเป็นคำแปลใหม่"
        )

        return "replace"

    if choice == "3":

        memory.append(
            new_item
        )

        print(
            "→ เก็บทั้งสองคำแปล"
        )

        return "keep_both"

    print(
        "→ ข้าม"
    )

    return "skip"


# ============================================================
# PROCESS ONE EXCEL
# ============================================================

def process_excel(
    path,
    memory
):

    filename = os.path.basename(
        path
    )

    print("\n")
    print("=" * 80)
    print(
        f"📖 อ่าน Excel: {filename}"
    )
    print("=" * 80)

    excel_data = read_excel(
        path
    )

    if not excel_data:

        print(
            "⚠️ Excel ไม่มีข้อมูล"
        )

        return {
            "new": 0,
            "duplicate": 0,
            "conflict": 0,
            "skip": 0
        }

    excel_text = excel_to_text(
        excel_data
    )

    print(
        f"\nส่งข้อมูลให้ {MODEL}..."
    )

    ai_items = ask_ollama(
        excel_text
    )

    print(
        f"AI พบข้อมูล: "
        f"{len(ai_items)} รายการ"
    )

    stats = {
        "new": 0,
        "duplicate": 0,
        "conflict": 0,
        "skip": 0
    }

    for raw_item in ai_items:

        item = validate_item(
            raw_item
        )

        if item is None:

            stats["skip"] += 1

            continue

        item["source_file"] = (
            filename
        )

        result = merge_item(
            memory,
            item
        )

        if result == "new":
            stats["new"] += 1

        elif result == "duplicate":
            stats["duplicate"] += 1

        elif result in {
            "keep_old",
            "replace",
            "keep_both"
        }:
            stats["conflict"] += 1

        else:
            stats["skip"] += 1

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        INPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        MEMORY_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("EXCEL → AI → TRANSLATION MEMORY")
    print("=" * 80)

    print(
        f"\nExcel folder:"
        f"\n{INPUT_DIR}\\"
    )

    print(
        f"\nMemory:"
        f"\n{MEMORY_FILE}"
    )

    # --------------------------------------------------------
    # Find Excel
    # --------------------------------------------------------

    files = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "*.xlsx"
            )
        )
    )

    if not files:

        print(
            "\n❌ ไม่พบไฟล์ Excel"
        )

        print(
            f"\nนำ .xlsx มาใส่ใน:"
        )

        print(
            f"{INPUT_DIR}\\"
        )

        return

    print(
        f"\nพบ Excel "
        f"{len(files)} ไฟล์"
    )

    # --------------------------------------------------------
    # Load Memory
    # --------------------------------------------------------

    memory = load_memory()

    print(
        f"Memory เดิม: "
        f"{len(memory):,} รายการ"
    )

    total = {
        "new": 0,
        "duplicate": 0,
        "conflict": 0,
        "skip": 0
    }

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for path in files:

        stats = process_excel(
            path,
            memory
        )

        for key in total:

            total[key] += stats[key]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_memory(
        memory
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("IMPORT COMPLETE")
    print("=" * 80)

    print(
        f"\n🟢 เพิ่มใหม่: "
        f"{total['new']:,}"
    )

    print(
        f"⚪ ซ้ำและคำแปลเหมือนเดิม: "
        f"{total['duplicate']:,}"
    )

    print(
        f"🟡 Conflict: "
        f"{total['conflict']:,}"
    )

    print(
        f"🔴 ข้าม: "
        f"{total['skip']:,}"
    )

    print(
        f"\nMemory ทั้งหมด: "
        f"{len(memory):,}"
    )

    print(
        f"\nบันทึก:"
        f"\n{MEMORY_FILE}"
    )

    print(
        f"\nIndex:"
        f"\n{MEMORY_INDEX_FILE}"
    )

    print(
        "\nเสร็จสิ้น"
    )


if __name__ == "__main__":
    main()