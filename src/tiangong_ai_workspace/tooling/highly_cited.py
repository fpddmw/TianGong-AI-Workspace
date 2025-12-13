import requests

TARGET_ISSN = "1879-0658"
TARGET_JOURNAL = "Resources, Conservation & Recycling"
PUBLISH_YEAR = "2023"


def _norm_journal_name(name) -> str:
    if not name:
        return ""
    name = name.lower().replace("&", "and")
    return " ".join("".join(ch if ch.isalnum() else " " for ch in name).split())


def is_target_journal(work: dict) -> bool:
    target = _norm_journal_name(TARGET_JOURNAL)
    for loc in work.get("locations", []) or []:
        source = (loc or {}).get("source") or {}
        if TARGET_ISSN in (source.get("issn") or []):
            if _norm_journal_name(source.get("display_name")) == target:
                return True
    return False


url = "https://api.openalex.org/works"
params = {
    "filter": f"publication_year:{PUBLISH_YEAR},locations.source.issn:{TARGET_ISSN}",
    "sort": "cited_by_count:desc",
    # "sort": "cited_by_count:asc",
    "per-page": 20,
    "select": "id,doi,title,publication_year,cited_by_count,type,locations",
}

r = requests.get(url, params=params, timeout=30)
r.raise_for_status()
# print(len(r.json().get("results", [])))
data = [w for w in r.json()["results"] if is_target_journal(w)]

for i, w in enumerate(data, 1):
    # print(i, w.get("cited_by_count"), w.get("type"), w.get("title"), w.get("doi"), w.get("id"))
    print(f'{i}, {w.get("cited_by_count")}, {w.get("title")}, {w.get("doi")}')
