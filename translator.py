import os
import re
import time
import json
import requests


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "qwen3:4b"

OLLAMA_URL = "http://localhost:11434/api/chat"

INPUT_DIR = "input"
OUTPUT_DIR = "output"

BATCH_SIZE = 8
CONTEXT_SIZE = 2

MAX_RETRIES = 3
SINGLE_RETRIES = 3

START_SUBTITLE = 1
END_SUBTITLE = 300

CONFIDENCE_THRESHOLD = 0.80
MAX_TRANSLATION_LENGTH = 120


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
คุณเป็นนักแปลบทละครจีน → ภาษาไทยระดับมืออาชีพ

หน้าที่:
แปลบทสนทนาภาษาจีนเป็นภาษาไทยให้เป็นธรรมชาติ เหมาะกับซับไตเติลละคร

กฎสำคัญ:

1. แปลภาษาจีนเป็นภาษาไทย
2. ห้ามตอบเป็นภาษาจีน
3. ห้ามแปลหมายเลข subtitle
4. ห้ามแปล timestamp
5. รักษาความหมายของต้นฉบับ
6. ใช้ภาษาไทยที่เป็นธรรมชาติ ไม่แปลตรงตัวจนแข็ง
7. รักษาบริบทของละคร
8. ชื่อบุคคลต้องใช้ให้สม่ำเสมอ
9. คำเรียก ตำแหน่ง ยศ และศัพท์เฉพาะต้องใช้ให้สม่ำเสมอ
10. ถ้ามีตัวละครหลายคน ให้รักษาผู้พูดตามบริบท
11. รักษาเครื่องหมายวรรคตอนที่จำเป็น
12. ถ้ามี [ ] หรือ ( ) ที่เป็นส่วนหนึ่งของบท ให้รักษาไว้ตามความเหมาะสม
13. ห้ามใส่คำอธิบายเพิ่มเติม
14. ห้ามใส่ Markdown
15. ห้ามใส่ timestamp
16. ห้ามใส่ subtitle number ใน translation
17. ห้ามแสดงความคิดหรือ reasoning
18. ให้ตอบเป็น JSON เท่านั้น

ความยาว:
- พยายามให้เหมาะกับ subtitle
- ไม่ควรยาวเกินประมาณ 40 ตัวอักษรต่อบรรทัด
- ถ้ายาวมาก สามารถแบ่งบรรทัดด้วย \\N ได้

สำหรับการแปล subtitle เดียว:
{
  "id": 1,
  "translation": "คำแปลภาษาไทย",
  "confidence": 0.95
}

สำหรับการแปลหลาย subtitle:
{
  "translations": [
    {
      "id": 1,
      "translation": "คำแปลภาษาไทย",
      "confidence": 0.95
    },
    {
      "id": 2,
      "translation": "คำแปลภาษาไทย",
      "confidence": 0.90
    }
  ]
}

confidence ต้องเป็นค่าระหว่าง 0 ถึง 1
"""


# ============================================================
# SRT PARSER
# ============================================================

def parse_srt(path):

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        content = f.read()

    blocks = re.split(
        r"\n\s*\n",
        content.strip()
    )

    subtitles = []

    for block in blocks:

        lines = block.strip().splitlines()

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

        subtitles.append({
            "index": index,
            "timecode": timecode,
            "source": text
        })

    return subtitles


# ============================================================
# EXISTING SRT
# ============================================================

def parse_existing_srt(path):

    if not os.path.exists(path):
        return {}

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        content = f.read()

    blocks = re.split(
        r"\n\s*\n",
        content.strip()
    )

    result = {}

    for block in blocks:

        lines = block.strip().splitlines()

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

        result[index] = {
            "index": index,
            "timecode": timecode,
            "translation": text
        }

    return result


# ============================================================
# SINGLE JSON PARSER
# ============================================================

def extract_translation(
    response_text,
    expected_id
):
    """
    อ่าน JSON จาก Ollama

    Expected:

    {
        "id": 1,
        "translation": "...",
        "confidence": 0.95
    }
    """

    try:

        data = json.loads(
            response_text
        )

        returned_id = int(
            data["id"]
        )

        if returned_id != expected_id:

            print(
                f"   ⚠️ ID mismatch: "
                f"expected {expected_id}, "
                f"got {returned_id}"
            )

            return None

        translation = str(
            data["translation"]
        ).strip()

        if not translation:

            print(
                "   ⚠️ Empty translation"
            )

            return None

        confidence = float(
            data.get(
                "confidence",
                0
            )
        )

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        return {
            "translation": translation,
            "confidence": confidence
        }

    except json.JSONDecodeError as e:

        print(
            f"   ⚠️ JSON parse error: {e}"
        )

        print(
            f"   RAW: {response_text}"
        )

        return None

    except Exception as e:

        print(
            f"   ⚠️ Parse error: {e}"
        )

        print(
            f"   RAW: {response_text}"
        )

        return None


# ============================================================
# BATCH JSON PARSER
# ============================================================

def extract_batch_translations(
    response_text,
    expected_ids
):
    """
    อ่าน JSON batch จาก Ollama

    Expected:

    {
        "translations": [
            {
                "id": 1,
                "translation": "...",
                "confidence": 0.95
            }
        ]
    }
    """

    try:

        data = json.loads(
            response_text
        )

        translations_list = data[
            "translations"
        ]

        if not isinstance(
            translations_list,
            list
        ):

            print(
                "   ⚠️ translations "
                "is not a list"
            )

            return None

        results = {}

        for item in translations_list:

            idx = int(
                item["id"]
            )

            translation = str(
                item["translation"]
            ).strip()

            confidence = float(
                item.get(
                    "confidence",
                    0
                )
            )

            confidence = max(
                0.0,
                min(1.0, confidence)
            )

            if not translation:

                print(
                    f"   ⚠️ Empty translation: "
                    f"{idx}"
                )

                return None

            # ตรวจ duplicate ID
            if idx in results:

                print(
                    f"   ⚠️ Duplicate ID: "
                    f"{idx}"
                )

                return None

            results[idx] = {
                "translation": translation,
                "confidence": confidence
            }

        expected_ids = set(
            expected_ids
        )

        returned_ids = set(
            results.keys()
        )

        # ตรวจ subtitle หาย
        missing_ids = (
            expected_ids
            - returned_ids
        )

        if missing_ids:

            print(
                f"   ⚠️ Missing IDs: "
                f"{sorted(missing_ids)}"
            )

            return None

        # ตรวจ subtitle เกิน
        extra_ids = (
            returned_ids
            - expected_ids
        )

        if extra_ids:

            print(
                f"   ⚠️ Unexpected IDs: "
                f"{sorted(extra_ids)}"
            )

            return None

        return results

    except json.JSONDecodeError as e:

        print(
            f"   ⚠️ Batch JSON parse error: "
            f"{e}"
        )

        print(
            f"   RAW:\n{response_text}"
        )

        return None

    except Exception as e:

        print(
            f"   ⚠️ Batch parse error: "
            f"{e}"
        )

        print(
            f"   RAW:\n{response_text}"
        )

        return None
# ============================================================
# QUALITY CHECKER
# ============================================================

def contains_chinese(text):
    """
    ตรวจว่ามีอักษรจีนหลงเหลืออยู่หรือไม่
    """

    if not text:
        return False

    chinese_pattern = re.compile(
        r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]"
    )

    return bool(
        chinese_pattern.search(text)
    )


def check_translation_quality(
    item,
    result
):
    """
    ตรวจคุณภาพคำแปลเบื้องต้น

    Return:
        {
            "needs_review": True/False,
            "reasons": [...]
        }
    """

    reasons = []

    if not result:

        reasons.append(
            "ไม่มีผลการแปล"
        )

        return {
            "needs_review": True,
            "reasons": reasons
        }

    translation = str(
        result.get(
            "translation",
            ""
        )
    ).strip()

    confidence = float(
        result.get(
            "confidence",
            0
        )
    )

    # --------------------------------------------------------
    # 1. Empty translation
    # --------------------------------------------------------

    if not translation:

        reasons.append(
            "คำแปลว่าง"
        )

    # --------------------------------------------------------
    # 2. Chinese remains
    # --------------------------------------------------------

    elif contains_chinese(
        translation
    ):

        reasons.append(
            "พบภาษาจีนหลงเหลือในคำแปล"
        )

    # --------------------------------------------------------
    # 3. Low confidence
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        reasons.append(
            f"confidence ต่ำ "
            f"({confidence:.2f})"
        )

    # --------------------------------------------------------
    # 4. Abnormally long
    # --------------------------------------------------------

    if len(translation) > MAX_TRANSLATION_LENGTH:

        reasons.append(
            f"คำแปลยาวผิดปกติ "
            f"({len(translation)} ตัวอักษร)"
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {
        "needs_review": len(reasons) > 0,
        "reasons": reasons
    }

# ============================================================
# ADD REVIEW RESULT
# ============================================================

def add_review_item(
    review_items,
    item,
    result,
    quality
):

    review_items.append({
        "index": item["index"],
        "timecode": item["timecode"],
        "source": item["source"],
        "translation": (
            result["translation"]
            if result
            else ""
        ),
        "confidence": (
            result["confidence"]
            if result
            else 0
        ),
        "reasons": quality[
            "reasons"
        ]
    })

# ============================================================
# WRITE REVIEW SRT
# ============================================================

def write_review_srt(
    path,
    review_items
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in review_items:

            f.write(
                f"{item['index']}\n"
            )

            f.write(
                f"{item['timecode']}\n"
            )

            # ใช้คำแปลก่อน
            # เพื่อให้เปิดดูใน player ได้
            translation = (
                item["translation"]
                if item["translation"]
                else "[ต้องตรวจสอบ]"
            )

            f.write(
                f"{translation}\n"
            )

            # เพิ่มข้อมูลสาเหตุในบรรทัดถัดไป
            f.write(
                f"[REVIEW: "
                f"{'; '.join(item['reasons'])}]\n\n"
            )    
# ============================================================
# CALL OLLAMA
# ============================================================

def call_ollama(
    prompt,
    num_predict=2048
):

    payload = {

        "model": MODEL_NAME,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        "stream": False,

        # ปิด thinking
        "think": False,

        # บังคับ JSON
        "format": "json",

        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": num_predict
        },

        "keep_alive": "10m"
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
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

        return content.strip()

    except requests.exceptions.RequestException as e:

        print(
            f"   ❌ Ollama request error: "
            f"{e}"
        )

        return None

    except Exception as e:

        print(
            f"   ❌ Ollama error: {e}"
        )

        return None


# ============================================================
# MAKE SINGLE PROMPT
# ============================================================

def make_prompt(
    item,
    previous_context=None
):

    prompt = ""

    if previous_context:

        prompt += (
            "บริบท subtitle ก่อนหน้า:\n"
        )

        for ctx in previous_context:

            prompt += (
                f"[{ctx['index']}] "
                f"{ctx['source']}\n"
            )

        prompt += "\n"

    prompt += (
        "แปล subtitle ต่อไปนี้:\n\n"
    )

    prompt += (
        f"[{item['index']}] "
        f"{item['source']}\n\n"
    )

    prompt += """
ตอบเป็น JSON เท่านั้น:

{
  "id": 1,
  "translation": "คำแปลภาษาไทย",
  "confidence": 0.95
}
"""

    return prompt


# ============================================================
# MAKE BATCH PROMPT
# ============================================================

def make_batch_prompt(
    items,
    previous_context=None
):

    prompt = ""

    if previous_context:

        prompt += (
            "บริบท subtitle ก่อนหน้า:\n"
        )

        for ctx in previous_context:

            prompt += (
                f"[{ctx['index']}] "
                f"{ctx['source']}\n"
            )

        prompt += "\n"

    prompt += (
        "แปล subtitle ต่อไปนี้ทั้งหมด:\n\n"
    )

    for item in items:

        prompt += (
            f"[{item['index']}] "
            f"{item['source']}\n"
        )

    prompt += """

ต้องแปล subtitle ทุกหมายเลข

ห้าม:
- ข้าม subtitle
- เพิ่ม subtitle
- รวม subtitle
- เปลี่ยน ID
- ใส่คำอธิบาย
- ใส่ Markdown
- ใส่ timestamp
- ใส่ข้อความนอก JSON

ตอบ JSON ตามรูปแบบนี้เท่านั้น:

{
  "translations": [
    {
      "id": 1,
      "translation": "คำแปลภาษาไทย",
      "confidence": 0.95
    },
    {
      "id": 2,
      "translation": "คำแปลภาษาไทย",
      "confidence": 0.90
    }
  ]
}
"""

    return prompt


# ============================================================
# TRANSLATE SINGLE
# ============================================================

def translate_single(
    item,
    previous_context=None
):

    for attempt in range(
        1,
        SINGLE_RETRIES + 1
    ):

        print(
            f"   🔄 Single attempt "
            f"{attempt}/{SINGLE_RETRIES}"
        )

        prompt = make_prompt(
            item,
            previous_context
        )

        response = call_ollama(
            prompt,
            num_predict=512
        )

        if not response:

            print(
                "   ⚠️ No response"
            )

            continue

        result = extract_translation(
            response,
            item["index"]
        )

        if result:

            print(
                f"   ✅ Translation: "
                f"{result['translation']}"
            )

            print(
                f"   🎯 Confidence: "
                f"{result['confidence']:.2f}"
            )

            return result

        print(
            "   ⚠️ Invalid translation result"
        )

        time.sleep(1)

    return None


# ============================================================
# TRANSLATE BATCH
# ============================================================

def translate_batch(
    items,
    previous_context=None
):

    expected_ids = [
        item["index"]
        for item in items
    ]

    prompt = make_batch_prompt(
        items,
        previous_context
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print(
            f"   🔄 Batch attempt "
            f"{attempt}/{MAX_RETRIES}"
        )

        response = call_ollama(
            prompt,
            num_predict=2048
        )

        if not response:

            print(
                "   ⚠️ No response"
            )

            continue

        results = extract_batch_translations(
            response,
            expected_ids
        )

        if results:

            print(
                f"   ✅ Batch OK: "
                f"{len(results)}/{len(items)}"
            )

            for idx in sorted(results):

                result = results[idx]

                print(
                    f"      [{idx}] "
                    f"{result['translation']} "
                    f"(conf="
                    f"{result['confidence']:.2f})"
                )

            return results

        print(
            "   ⚠️ Invalid batch result"
        )

        time.sleep(1)

    return None


# ============================================================
# WRITE SRT
# ============================================================

def write_srt(
    path,
    subtitles,
    translations
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in subtitles:

            idx = item["index"]

            result = translations.get(
                idx
            )

            if isinstance(
                result,
                dict
            ):

                translation = result[
                    "translation"
                ]

            else:

                translation = result

            if not translation:

                translation = item[
                    "source"
                ]

            f.write(
                f"{idx}\n"
                f"{item['timecode']}\n"
                f"{translation}\n\n"
            )


# ============================================================
# FIND INPUT FILE
# ============================================================

def find_input_file():

    if not os.path.exists(
        INPUT_DIR
    ):

        os.makedirs(
            INPUT_DIR
        )

    files = [
        f
        for f in os.listdir(
            INPUT_DIR
        )
        if f.lower().endswith(".srt")
    ]

    if not files:

        print(
            f"❌ ไม่พบไฟล์ .srt "
            f"ใน {INPUT_DIR}/"
        )

        return None

    if len(files) == 1:

        return os.path.join(
            INPUT_DIR,
            files[0]
        )

    print(
        "\nพบไฟล์ SRT หลายไฟล์:"
    )

    for i, filename in enumerate(
        files,
        1
    ):

        print(
            f"{i}. {filename}"
        )

    while True:

        try:

            choice = int(
                input("เลือกไฟล์: ")
            )

            if (
                1 <= choice
                <= len(files)
            ):

                return os.path.join(
                    INPUT_DIR,
                    files[choice - 1]
                )

        except ValueError:

            pass

        print(
            "❌ กรุณาเลือกหมายเลข "
            "ให้ถูกต้อง"
        )


# ============================================================
# TEST SINGLE
# ============================================================

def test_single_translation():

    print("\n")

    print("=" * 60)

    print(
        "TEST SINGLE SUBTITLE"
    )

    print("=" * 60)

    item = {
        "index": 1,
        "source": "你好，你是谁？"
    }

    print(
        f"\n🇨🇳 Source:\n"
        f"{item['source']}"
    )

    result = translate_single(
        item
    )

    print(
        "\n" + "-" * 60
    )

    if result:

        print(
            "✅ RESULT"
        )

        print(
            f"ID: {item['index']}"
        )

        print(
            f"Translation: "
            f"{result['translation']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']}"
        )

    else:

        print(
            "❌ Translation failed"
        )

    print(
        "-" * 60
    )


# ============================================================
# TEST BATCH
# ============================================================

def test_batch_translation():

    print("\n")

    print("=" * 60)

    print(
        "TEST BATCH TRANSLATION"
    )

    print("=" * 60)

    items = [

        {
            "index": 1,
            "source": "你好，你是谁？"
        },

        {
            "index": 2,
            "source": "我是李明。"
        },

        {
            "index": 3,
            "source": "你为什么来这里？"
        },

        {
            "index": 4,
            "source": "我来找一个人。"
        },

        {
            "index": 5,
            "source": "找谁？"
        },

        {
            "index": 6,
            "source": "这不关你的事。"
        },

        {
            "index": 7,
            "source": "我们走吧。"
        },

        {
            "index": 8,
            "source": "等等。"
        }
    ]

    results = translate_batch(
        items
    )

    print(
        "\n" + "=" * 60
    )

    if results:

        print(
            "✅ BATCH SUCCESS"
        )

        for idx in sorted(
            results
        ):

            result = results[idx]

            print(
                f"[{idx}] "
                f"{result['translation']} "
                f"(confidence="
                f"{result['confidence']:.2f})"
            )

    else:

        print(
            "❌ BATCH FAILED"
        )

    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    input_file = find_input_file()

    if not input_file:
        return

    print(
        f"\n📄 Input: "
        f"{input_file}"
    )

    subtitles = parse_srt(
        input_file
    )

    print(
        f"📊 Total subtitles: "
        f"{len(subtitles)}"
    )

    subtitles = [
        item
        for item in subtitles
        if START_SUBTITLE
        <= item["index"]
        <= END_SUBTITLE
    ]

    print(
        f"🎯 Processing: "
        f"{len(subtitles)} subtitles"
    )

    translations = {}

    review_items = []

    # ========================================================
    # TRANSLATION
    # ========================================================

    for i in range(
        0,
        len(subtitles),
        BATCH_SIZE
    ):

        batch = subtitles[
            i:i + BATCH_SIZE
        ]

        print(
            f"\n🔄 Batch "
            f"{i + 1}-"
            f"{i + len(batch)}"
        )

        result = translate_batch(
            batch
        )

        # ----------------------------------------------------
        # Batch success
        # ----------------------------------------------------

        if result:

            for item in batch:

                idx = item[
                    "index"
                ]

                translation_result = (
                    result.get(idx)
                )

                translations[idx] = (
                    translation_result
                )

                quality = (
                    check_translation_quality(
                        item,
                        translation_result
                    )
                )

                if quality[
                    "needs_review"
                ]:

                    add_review_item(
                        review_items,
                        item,
                        translation_result,
                        quality
                    )

                    print(
                        f"   ⚠️ REVIEW "
                        f"[{idx}] "
                        f"{', '.join(quality['reasons'])}"
                    )

        # ----------------------------------------------------
        # Batch failed → Single
        # ----------------------------------------------------

        else:

            print(
                "⚠️ Batch failed"
            )

            print(
                "🔄 Falling back "
                "to single"
            )

            for item in batch:

                translation_result = (
                    translate_single(
                        item
                    )
                )

                idx = item[
                    "index"
                ]

                if translation_result:

                    translations[idx] = (
                        translation_result
                    )

                quality = (
                    check_translation_quality(
                        item,
                        translation_result
                    )
                )

                if quality[
                    "needs_review"
                ]:

                    add_review_item(
                        review_items,
                        item,
                        translation_result,
                        quality
                    )

                    print(
                        f"   ⚠️ REVIEW "
                        f"[{idx}] "
                        f"{', '.join(quality['reasons'])}"
                    )

    # ========================================================
    # WRITE TRANSLATED SRT
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        "translated.srt"
    )

    write_srt(
        output_path,
        subtitles,
        translations
    )

    # ========================================================
    # WRITE REVIEW SRT
    # ========================================================

    review_path = os.path.join(
        OUTPUT_DIR,
        "review.srt"
    )

    write_review_srt(
        review_path,
        review_items
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("TRANSLATION COMPLETE")
    print("=" * 60)

    print(
        f"Total: "
        f"{len(subtitles)}"
    )

    print(
        f"Translated: "
        f"{len(translations)}"
    )

    print(
        f"Need review: "
        f"{len(review_items)}"
    )

    print(
        f"\n✅ Translation:"
        f"\n{output_path}"
    )

    print(
        f"\n⚠️ Review:"
        f"\n{review_path}"
    )

    print(
        "=" * 60
    )

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

