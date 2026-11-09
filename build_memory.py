# -*- coding: utf-8 -*-
"""
เทียบข้อมูล input (Chinese) กับ output (Thai) ออกเป็น JSON
พร้อมตัดคำเชื่อม (stopwords) ทิ้ง เหลือแต้คำสำคัญไว้ใช้เป็น Translation Memory

ใช้:
    python build_memory.py                       # ทุกไฟล์ใน input/
    python build_memory.py ex0.srt ex1.srt       # เฉพาะที่ระบุ
    python build_memory.py --min-len=20          # กรอง keyword รวมยาว >= 20

ผลลัพธ:
    comparison/memory/translation_memory.jsonl  (เข้ากบั format เดิมของ translator.py)
    comparison/memory/translation_memory.json   (ตวั เต็ม + metadata)
"""

import os
import re
import sys
import json
from collections import Counter

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = "input"     # Chinese (zh)
OUTPUT_DIR = "output"   # Thai (th)
MEMORY_DIR = os.path.join("comparison", "memory")

OUTPUT_JSONL = os.path.join(MEMORY_DIR, "translation_memory.jsonl")
OUTPUT_JSON = os.path.join(MEMORY_DIR, "translation_memory.json")

# ============================================================
# STOPWORDS (คำเชื่อม / คำบ่อย ที่ไม่ใช้คำสำคัญ)
# ============================================================

ZH_STOPWORDS = sorted(
    [
        # particles / structural
        "的", "了", "着", "过", "地", "得", "吗", "呢", "吧",
        "啊", "呀", "哦", "嗯", "嘛", "么", "罢了", "而已",
        # pronouns
        "我", "我们", "你", "你们", "您", "他", "她", "它",
        "他们", "她们", "咱们", "自己", "别人", "人家",
        # common verbs / auxiliaries / negation
        "是", "在", "有", "没", "没有", "不", "不会", "不要",
        "别", "又", "还", "就", "都", "也", "才", "再",
        "会", "能", "能够", "可以", "要", "想", "应该",
        "必须", "把", "被", "让", "给", "对", "跟", "和",
        "与", "或", "或者", "但", "但是", "可是", "不过",
        "因为", "所以", "如果", "虽然", "而", "且",
        # question / demonstrative
        "这", "那", "这个", "那个", "这些", "那些",
        "什么", "谁", "哪", "哪儿", "哪里", "怎么", "怎样",
        "为什么", "多少", "几", "怎",
        # quantifiers / numbers
        "一", "一个", "些", "个", "点", "有点",
        # time / filler
        "现在", "已经", "正在", "刚才", "然后",
        "其实", "当然", "真的", "反正", "总之",
        # interjections
        "哎", "喂", "嘿", "哈", "哼", "切",
    ],
    key=len,
    reverse=True,
)

ZH_STOPSET = set(ZH_STOPWORDS)

TH_STOPWORDS = {
    # คำเชื่อม/คำบ่อย (รวม variant ที่พบบ่อยในโปรเจกต์นี้)
    "ที่", "ที่", "ก็", "ก็", "ได้", "ได้", "เป็น", "เป็น",
    "จะ", "ไม่", "ไม", "ใน", "บน", "กับ", "กบั", "และ",
    "หรือ", "หรือ", "อย่าง", "อย่าง", "เช่น", "เช่น",
    "แต่", "แต้", "เพราะ", "เหตু", "เหตุ", "น้ี", "นี้",
    "นั้น", "นั้ น", "นะคะ", "นะ", "นะคะ", "หรอก", "อิก",
    "มาก", "มาย", "น้อย", "น้อย", "ด้าย", "มี", "มื",
    "อยู่", "อยู่", "ไป", "มา", "บอก", "ทำ", "ทํา",
    "เห็น", "เห็น", "รู้", "ร้้", "เข้าใจ", "เข้าใจ",
    "จริง", "จิง", "แล้ว", "แลว", "กำลัง", "กําลัง",
    "ยัง", "เธอ", "เขา", "มัน", "เรา", "คุณ", "คณุ",
    "ผม", "ดิ", "ชั้น", "ชั้ น", "ตัวเอง", "ตวั เอง",
    "อัน", "อะไร", "อะไร", "อย่าง", "อย่างน้ี",
}

# ============================================================
# SRT PARSER
# ============================================================

def parse_srt(path):

    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    blocks = re.split(r"\n\s*\n", raw.strip())

    items = []

    for block in blocks:

        lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip()
        ]

        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        text = " ".join(lines[2:]).strip()

        items.append(
            {
                "index": index,
                "text": text,
            }
        )

    return items


# ============================================================
# NORMALIZE
# ============================================================

def normalize(text):

    # รวม \\N และ newline เป็น space
    text = text.replace("\\N", " ")
    text = re.sub(r"\s+", " ", text)

    # ตัดชื่อบุคคลใน [ ] และ ( ) เพราะทำให้ pair ไม่ตรง
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)

    # ตัดเครื่องหมายวรรคตอน
    text = re.sub(
        r"[，。！？；：、,.!?;:～~…\"'“”‘’]+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# KEYWORD EXTRACTION (ตัดคำเชื่อมทิ้ง เหลือคำสำคัญ)
# ============================================================

def zh_keywords(text):



# ============================================================
# PAIR BUILDER (เทียบ input zh <-> output th)
# ============================================================

def load_pairs(input_dir, output_dir, wanted=None):

    pairs = []

    files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith(".srt")
    )

    if wanted:
        files = [f for f in files if f in wanted]

    for name in files:

        in_path = os.path.join(input_dir, name)

        # output/ บางไฟล์ชื่อปน (ext1.srt vs ex1.srt)
        out_path = os.path.join(output_dir, name)

        if not os.path.exists(out_path):
            alt = os.path.join(
                output_dir,
                name.replace("ex", "ext", 1),
            )
            if os.path.exists(alt):
                out_path = alt
            else:
                print(f"⚠️ skip (no output match): {name}")
                continue

        zh_items = {
            it["index"]: it["text"]
            for it in parse_srt(in_path)
        }

        th_items = {
            it["index"]: it["text"]
            for it in parse_srt(out_path)
        }

        matched = 0

        for idx in sorted(set(zh_items) & set(th_items)):

            zh_raw = zh_items[idx]
            th_raw = th_items[idx]

            zh_norm = normalize(zh_raw)
            th_norm = normalize(th_raw)

            if not zh_norm or not th_norm:
                continue

            matched += 1

            pairs.append(
                {
                    "file": name,
                    "index": idx,
                    "source": zh_norm,
                    "target": th_norm,
                }
            )

        print(
            f"   {name}: "
            f"zh={len(zh_items)} th={len(th_items)} "
            f"matched={matched}"
        )

    return pairs


# ============================================================
# MEMORY BUILDER
# ============================================================

def build_memory(pairs, min_len=0):

    counter = Counter(
        (p["source"], p["target"])
        for p in pairs
    )

    entries = []

    for (source, target), count in counter.most_common():

        zh_words = zh_keywords(source)
        th_words = th_keywords(target)

        kw = make_keywords(zh_words, th_words)

        if len(kw["combined"]) < min_len:
            continue

        entries.append(
            {
                "source": source,
                "target": target,
                "keywords": kw,
                "category": guess_category(zh_words),
                "count": count,
            }
        )

    return entries


# ============================================================
# GLOBAL IDF (ตัดคำที่โผล่เกือบทุกบรรทัด = คำเชื่อมแฝง)
# ============================================================

def apply_idf(entries, threshold=0.20):

    total = len(entries)

    if not total:
        return entries

    doc_freq = Counter()

    for e in entries:
        for w in set(e["keywords"]["zh"].split()):
            doc_freq[w] += 1

    drop = {
        w
        for w, df in doc_freq.items()
        if df / total >= threshold
    }

    if drop:
        print(
            "🚫 auto-drop stopwords "
            f"(df >= {threshold:.0%}): {' '.join(sorted(drop))}"
        )

    for e in entries:

        zh = [
            w
            for w in e["keywords"]["zh"].split()
            if w not in drop
        ]

        e["keywords"]["zh"] = " ".join(zh)
        e["keywords"]["combined"] = (
            f"{e['keywords']['zh']} | {e['keywords']['th']}"
        )

    return entries


    remaining = text

    # ลบ stopwords (ยาวปกอน) แล้วเสียบ space กันคำติดกัน
    for sw in ZH_STOPWORDS:
        remaining = remaining.replace(sw, " ")

    # เก็บอักษรจีนที่เหลือ
    words = re.findall(r"[\u4e00-\u9fff]+", remaining)

    words = [w for w in words if w not in ZH_STOPSET]

    return words


def th_keywords(text):

    tokens = re.findall(
        r"[\u0e00-\u0e7fa-zA-Z0-9]+",
        text,
    )

    return [
        t
        for t in tokens
        if t not in TH_STOPWORDS and len(t) >= 2
    ]


def make_keywords(zh_words, th_words):

    zh = " ".join(zh_words)
    th = " ".join(th_words)

    return {
        "zh": zh,
        "th": th,
        "combined": f"{zh} | {th}",
    }


def guess_category(zh_words):

    joined = "".join(zh_words)

    if len(joined) <= 2:
        return "term"

    if len(joined) <= 4:
        return "phrase"

    return "sentence"



