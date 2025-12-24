from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional

import httpx
import numpy as np

__all__ = ["JournalCitationBands", "fetch_journal_citation_bands"]


@dataclass(slots=True)
class JournalCitationBands:
    high: list[Mapping[str, object]]
    middle: list[Mapping[str, object]]
    low: list[Mapping[str, object]]
    thresholds: Mapping[str, float]


def _norm_journal_name(name: str) -> str:
    name = name.lower().replace("&", "and")
    return " ".join("".join(ch if ch.isalnum() else " " for ch in name).split())


def _is_target_journal(work: Mapping[str, object], target_journal: str, target_issn: str) -> bool:
    target = _norm_journal_name(target_journal)
    for loc in work.get("locations", []) or []:
        if not isinstance(loc, Mapping):
            continue
        source = (loc or {}).get("source") or {}
        if not isinstance(source, Mapping):
            continue
        issns = source.get("issn") or []
        if not isinstance(issns, Iterable):
            continue
        if target_issn in issns:
            display = source.get("display_name")
            if isinstance(display, str) and _norm_journal_name(display) == target:
                return True
    return False


def _is_valid_work(work: Mapping[str, object]) -> bool:
    title = work.get("title")
    if not isinstance(title, str):
        return False
    normalized_title = title.strip().lower()
    if normalized_title.startswith("corrigendum to"):
        return False
    if normalized_title.startswith("editorial board"):
        return False
    doi = work.get("doi")
    return isinstance(doi, str) and bool(doi.strip())


def _fetch_all_works(journal_issn: str, journal_name: str, publish_year: Optional[int], max_records: int = 800) -> list[Mapping[str, object]]:
    """Fetch works for a journal, sorted by citations descending."""

    params: MutableMapping[str, object] = {
        "filter": f"locations.source.issn:{journal_issn}",
        "sort": "cited_by_count:desc",
        "per-page": 200,
        "select": "id,doi,title,publication_year,cited_by_count,referenced_works_count,abstract_inverted_index,open_access,type,locations",
        "cursor": "*",
    }
    if publish_year:
        params["filter"] = f"{params['filter']},publication_year:{publish_year}"

    url = "https://api.openalex.org/works"
    works: list[Mapping[str, object]] = []
    while len(works) < max_records:
        response = httpx.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            break
        for work in results:
            if _is_target_journal(work, journal_name, journal_issn) and _is_valid_work(work):
                works.append(work)
                if len(works) >= max_records:
                    break
        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
    return works


def fetch_journal_citation_bands(
    *,
    journal_issn: str,
    journal_name: str,
    publish_year: Optional[int] = None,
    band_limit: int = 200,
    max_records: int = 800,
) -> JournalCitationBands:
    """
    Return high/middle/low citation bands for a journal.

    High/Low bands are determined by citation percentiles (>=75% is high, <=25% is low).
    Results within each band are sorted by citation count and truncated to `band_limit`.
    """

    works = _fetch_all_works(journal_issn, journal_name, publish_year, max_records)
    works_sorted = sorted(works, key=lambda w: int(w.get("cited_by_count") or 0), reverse=True)
    total = len(works_sorted)
    if total == 0:
        return JournalCitationBands(high=[], middle=[], low=[], thresholds={"p25": 0.0, "p75": 0.0})

    citations = np.array([int(w.get("cited_by_count") or 0) for w in works_sorted])
    p25 = float(np.percentile(citations, 25))
    p75 = float(np.percentile(citations, 75))

    high = [w for w in works_sorted if int(w.get("cited_by_count") or 0) >= p75][:band_limit]
    low = [w for w in reversed(works_sorted) if int(w.get("cited_by_count") or 0) <= p25][:band_limit]
    low = list(reversed(low))
    middle_candidates = [w for w in works_sorted if p25 < int(w.get("cited_by_count") or 0) < p75]
    middle = middle_candidates[:band_limit]

    return JournalCitationBands(high=high, middle=middle, low=low, thresholds={"p25": p25, "p75": p75})
