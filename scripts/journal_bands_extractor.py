#!/usr/bin/env python3
"""
脚本：从 journal_bands.md 提取 journal band analysis 结果并汇总为表格与统计。

用法:
  python scripts/journal_bands_extractor.py
  python scripts/journal_bands_extractor.py -i journal_bands.md -o out.csv

输出：CSV (`out.csv`)、JSON (`out.json`)，并在终端打印聚合统计。
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from statistics import mean, median, pstdev

EVAL_MAP = {"优": 3, "中": 2, "差": 1}


def extract_records(md_text: str):
    lines = md_text.splitlines()
    records = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"- \*\*(.*?)\*\* \((\d{4}), cites: (\d+)\)", line)
        if m:
            title = m.group(1).strip()
            year = int(m.group(2))
            citations = int(m.group(3))
            # default
            authors = ""
            band = ""
            rule_suggestion = ""
            dims = {
                "topic": {"eval": "", "reason": ""},
                "methodology": {"eval": "", "reason": ""},
                "data": {"eval": "", "reason": ""},
                "impact": {"eval": "", "reason": ""},
                "presentation": {"eval": "", "reason": ""},
            }

            # scan forward to find the JSON blob that follows "原始 LLM 输出:" or a line with '{"citation_band"'
            j = i + 1
            json_text = None
            while j < len(lines):
                line_value = lines[j]
                if '{"citation_band"' in line_value or line_value.strip().startswith('{"citation_band"'):
                    start = j
                    brace = 0
                    acc = []
                    k = start
                    while k < len(lines):
                        acc.append(lines[k])
                        brace += lines[k].count("{") - lines[k].count("}")
                        if brace == 0:
                            break
                        k += 1
                    json_text = "\n".join(acc)
                    i = k
                    break
                if lines[j].lstrip().startswith("- {") or lines[j].lstrip().startswith('- {"citation_band"') or lines[j].lstrip().startswith("    - {"):
                    start = j
                    brace = 0
                    acc = []
                    k = start
                    while k < len(lines):
                        acc.append(lines[k].lstrip().lstrip("- ").rstrip())
                        brace += lines[k].count("{") - lines[k].count("}")
                        if brace == 0 and acc:
                            break
                        k += 1
                    json_text = "\n".join(acc)
                    i = k
                    break
                j += 1

            if json_text:
                try:
                    parsed = json.loads(json_text)
                except Exception:
                    try:
                        cleaned = re.sub(r"^\s*-\s*", "", json_text)
                        parsed = json.loads(cleaned)
                    except Exception:
                        parsed = None

                if isinstance(parsed, dict):
                    band = parsed.get("citation_band", parsed.get("citationBand", ""))
                    rule_suggestion = parsed.get("rule_suggestion", parsed.get("ruleSuggestion", ""))
                    features = parsed.get("features_analysis") or parsed.get("features") or {}
                    for dim in ["topic", "methodology", "data", "impact", "presentation"]:
                        info = features.get(dim, {})
                        if isinstance(info, dict):
                            dims[dim]["eval"] = info.get("eval", info.get("evaluation", ""))
                            dims[dim]["reason"] = info.get("reason", info.get("explanation", ""))
            records.append(
                {
                    "Title": title,
                    "Authors": authors,
                    "Year": year,
                    "Citations": citations,
                    "Band": band,
                    "Features": dims,
                    "rule_suggestion": rule_suggestion,
                }
            )
        i += 1
    return records


def score_and_flatten(records):
    rows = []
    for r in records:
        dims = r["Features"]
        scores = {}
        total = 0
        for dim in ["topic", "methodology", "data", "impact", "presentation"]:
            eval_text = dims[dim].get("eval", "")
            score = EVAL_MAP.get(eval_text, None)
            if score is None:
                score = {"good": 3, "medium": 2, "poor": 1}.get(eval_text.lower(), 0)
            scores[dim] = score
            total += score

        row = {
            "Title": r["Title"],
            "Authors": r["Authors"],
            "Year": r["Year"],
            "Citations": r["Citations"],
            "Band": r["Band"],
            "Topic_score": scores["topic"],
            "Methodology_score": scores["methodology"],
            "Data_score": scores["data"],
            "Impact_score": scores["impact"],
            "Presentation_score": scores["presentation"],
            "Topic_reason": dims["topic"].get("reason", ""),
            "Methodology_reason": dims["methodology"].get("reason", ""),
            "Data_reason": dims["data"].get("reason", ""),
            "Impact_reason": dims["impact"].get("reason", ""),
            "Presentation_reason": dims["presentation"].get("reason", ""),
            "rule_suggestion": r.get("rule_suggestion", ""),
            "total_score": total,
        }
        rows.append(row)
    return rows


def compute_stats(rows):
    stats = {}
    totals = [r["total_score"] for r in rows]
    stats["n"] = len(rows)
    stats["total_mean"] = mean(totals) if totals else 0
    stats["total_median"] = median(totals) if totals else 0
    stats["total_stdpop"] = pstdev(totals) if len(totals) > 1 else 0

    by_band = defaultdict(list)
    for r in rows:
        by_band[r["Band"]].append(r["total_score"])

    stats["by_band"] = {}
    for b, arr in by_band.items():
        stats["by_band"][b] = {
            "n": len(arr),
            "mean": mean(arr) if arr else 0,
            "median": median(arr) if arr else 0,
            "stdpop": pstdev(arr) if len(arr) > 1 else 0,
        }

    dims = ["Topic_score", "Methodology_score", "Data_score", "Impact_score", "Presentation_score"]
    stats["dimensions"] = {d: {} for d in dims}
    for d in dims:
        vals = [r[d] for r in rows if isinstance(r[d], (int, float))]
        stats["dimensions"][d]["mean"] = mean(vals) if vals else 0
    stats["dimensions_by_band"] = {}
    for b, arr in by_band.items():
        members = [r for r in rows if r["Band"] == b]
        stats["dimensions_by_band"][b] = {}
        for d in dims:
            vals = [m[d] for m in members if isinstance(m[d], (int, float))]
            stats["dimensions_by_band"][b][d] = mean(vals) if vals else 0

    return stats


def write_outputs(rows, out_csv, out_json):
    fieldnames = [
        "Title",
        "Authors",
        "Year",
        "Citations",
        "Band",
        "Topic_score",
        "Methodology_score",
        "Data_score",
        "Impact_score",
        "Presentation_score",
        "total_score",
        "Topic_reason",
        "Methodology_reason",
        "Data_reason",
        "Impact_reason",
        "Presentation_reason",
        "rule_suggestion",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    with open(out_json, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", default="journal_bands_with_citation.md")
    p.add_argument("-o", "--out", default="scripts/out.csv")
    p.add_argument("-j", "--json", default="scripts/out.json")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        md = f.read()

    records = extract_records(md)
    rows = score_and_flatten(records)
    stats = compute_stats(rows)

    write_outputs(rows, args.out, args.json)

    print(f"Wrote {len(rows)} records to {args.out} and {args.json}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
