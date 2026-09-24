"""Build the English filler corpus for P5.

The French filler is real administrative prose on property and gifts, supplied through
FR_CORPUS (see README). This fetches an English counterpart of comparable register and
subject so the FR/EN comparison is not confounded by topic.

Wikipedia plain-text extracts, cached to disk. Run once.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from _common import ROOT

TITLES = [
    "Mortgage loan", "Real estate", "Property law", "Inheritance tax",
    "Stamp duty", "Mortgage-backed security", "Real estate appraisal",
    "Landlord", "Leasehold estate", "Conveyancing", "Home equity loan",
    "Loan", "Credit score", "Debt-to-income ratio", "Underwriting",
    "Foreclosure", "Property tax", "Estate planning", "Trust law",
    "Gift tax in the United States",
]

OUT = ROOT / "data" / "en-corpus.txt"
API = "https://en.wikipedia.org/w/api.php"


def extract(title: str) -> str:
    # curl rather than urllib: urllib gets a hard 429 from this machine's egress, curl
    # does not. Not worth diagnosing further, the corpus is fetched once.
    query = urllib.parse.urlencode(
        {"action": "query", "prop": "extracts", "explaintext": "1",
         "format": "json", "redirects": "1", "titles": title}
    )
    raw = subprocess.run(
        ["curl", "-sS", "-A", "typed-decision-bench/0.1 (research)", f"{API}?{query}"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    pages = json.loads(raw)["query"]["pages"]
    return next(iter(pages.values())).get("extract", "")


def main() -> None:
    # Only ~130k chars are needed (the measured input ceiling). Tolerate the occasional
    # throttled response rather than failing the whole fetch.
    parts: list[str] = []
    for title in TITLES:
        try:
            text = extract(title)
        except Exception as exc:
            print(f"{'skip':>8}  {title}  ({type(exc).__name__})")
            continue
        print(f"{len(text):>8,}  {title}")
        if text:
            parts.append(text)
        if sum(map(len, parts)) > 200_000:
            break
        time.sleep(1.0)

    OUT.parent.mkdir(exist_ok=True)
    corpus = "\n\n".join(parts)
    OUT.write_text(corpus, encoding="utf-8")
    print(f"\nwrote {OUT}  ({len(corpus):,} chars)")


if __name__ == "__main__":
    main()
