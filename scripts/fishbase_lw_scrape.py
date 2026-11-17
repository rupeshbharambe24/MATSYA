#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FishBase length–weight scraper (robust, no <table> dependency)

Pipeline:
  1) Load the LW list page: https://www.fishbase.org/topic/List.php?group=lw
  2) Collect all species LW pages (LWRelationshipList.php?...), dedupe by Species ID.
  3) For each species LW page, collect ALL per-study detail links (FishLWSummary.php?...).
  4) Visit each detail page and parse labeled fields.
  5) Save a tidy Excel with:
       - All rows in All_LW
       - An Index sheet (counts per species)

Notes:
  - Auto-fallback to fishbase.se / fishbase.de mirrors.
  - Polite delay between requests.
  - Resilient parsing (label mapping + whitespace normalization).

Usage:
  python scripts/fishbase_lw_scrape.py \
    --start-url "https://www.fishbase.org/topic/List.php?group=lw" \
    --out "assets/fishbase_lw_raw.xlsx" \
    --delay 1.0 \
    --max-species 0     # 0 = all species from list page

"""

import argparse
import re
import sys
import time
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import pandas as pd

MIRRORS = [
    "https://www.fishbase.org",
    "https://www.fishbase.se",
    "https://www.fishbase.de",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}

def build_session(timeout: float) -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    retries = Retry(
        total=5, connect=3, read=3, backoff_factor=0.6,
        status_forcelist=[403, 408, 429, 500, 502, 503, 504],
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))

    orig_request = s.request
    def _with_default_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return orig_request(method, url, **kwargs)
    s.request = _with_default_timeout
    return s

def get_with_mirror(session: requests.Session, url: str):
    """Try original host; if fishbase.* and fails, try mirrors. Return (response, final_url)."""
    trials = []
    parsed = urlparse(url)
    if any(h in parsed.netloc for h in ["fishbase.org", "fishbase.se", "fishbase.de"]):
        trials.append(url)
        path_q = parsed.path + (("?" + parsed.query) if parsed.query else "")
        for m in MIRRORS:
            cand = m + path_q
            if cand not in trials:
                trials.append(cand)
    else:
        trials.append(url)

    last_e = None
    for cand in trials:
        try:
            r = session.get(cand)
            if r.status_code == 200:
                return r, cand
        except requests.RequestException as e:
            last_e = e
    if last_e:
        raise last_e
    raise RuntimeError(f"Failed all mirrors for {url}")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# ---------- Step 1: collect species LW pages from the list ----------
def collect_species_lw_urls(session, list_url: str, delay: float) -> dict:
    """
    Return dict {SpeciesID: absolute LWRelationshipList.php URL}.
    Handles the entire paginated list by iterating start=0,500,1000,...
    Falls back to classic pagination if the total can't be detected.
    """
    import re
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    def _gather_from_page(url, species_dict):
        resp, final = get_with_mirror(session, url)
        time.sleep(delay)
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "LWRelationshipList.php" not in href:
                continue
            absurl = urljoin(final, href)
            sp_id = parse_qs(urlparse(absurl).query).get("ID", [None])[0]
            if sp_id and str(sp_id).isdigit():
                species_dict.setdefault(int(sp_id), absurl)
        return resp.text  # so we can parse "( 1 to 500 of 7,371 )"

    # 1) Hit the first page, capture total, and gather species
    species = {}
    first_html = _gather_from_page(list_url, species)

    # Try to parse the total count from the header: "( 1 to 500 of 7,371 )"
    m = re.search(r"\(\s*\d+\s*to\s*\d+\s*of\s*([\d,]+)\s*\)", first_html, re.I)
    if m:
        total = int(m.group(1).replace(",", ""))
        # 2) Build all start offsets in steps of 500 (first page was already fetched)
        # Normalize the URL and set start param for subsequent pages
        parsed = urlparse(list_url)
        base_q = parse_qs(parsed.query)
        base_q.pop("start", None)  # ensure we control it
        for start in range(500, total, 500):
            q = {k: v[0] if isinstance(v, list) and v else v for k, v in base_q.items()}
            q["start"] = str(start)
            new_query = urlencode(q)
            page_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", new_query, ""))
            _gather_from_page(page_url, species)
    else:
        # Fallback: old logic — follow number links if we can’t detect the total
        # (usually unnecessary, but keeps behavior robust)
        to_visit, visited = [list_url], set()
        while to_visit:
            url = to_visit.pop(0)
            if url in visited:
                continue
            visited.add(url)
            resp, final = get_with_mirror(session, url)
            time.sleep(delay)
            soup = BeautifulSoup(resp.text, "html.parser")
            _ = _gather_from_page(url, species)
            # enqueue numeric page links if any
            for a in soup.find_all("a", href=True):
                txt = (a.string or "").strip()
                if txt.isdigit():
                    to_visit.append(urljoin(final, a["href"]))

    return species


    def _collect_from_list_soup(soup, base_url, species_dict):
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "LWRelationshipList.php" not in href:
                continue
            absurl = urljoin(base_url, href)
            sp_id = parse_qs(urlparse(absurl).query).get("ID", [None])[0]
            if sp_id and sp_id.isdigit():
                species_dict.setdefault(int(sp_id), absurl)

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        next_url = _scrape_one(url)
        if next_url:
            to_visit.append(next_url)

    return species

# ---------- Step 2: per-species page -> collect all detail links ----------
def collect_detail_links_from_species(session, lw_url: str, delay: float) -> list:
    """
    On LWRelationshipList.php, each study row has the 'a' value as a link to FishLWSummary.php.
    Return a list of absolute FishLWSummary URLs.
    """
    resp, final = get_with_mirror(session, lw_url)
    time.sleep(delay)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "FishLWSummary.php" in href:
            links.append(urljoin(final, href))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# ---------- Step 3: parse one detail page ----------
LABEL_MAP = {
    "a :": "a",
    "b :": "b",
    "sex :": "Sex",
    "number of fish :": "n",
    "length (cm) :": "Length (cm)",
    "r^{2} :": "r2",
    "territory :": "Territory",
    "locality :": "Locality",
    "estimate doubtful ? :": "Doubtful?",
    "method :": "Method",
    "main ref. :": "MainRef",
    "data ref. :": "DataRef",
}

def parse_detail(session, detail_url: str, delay: float) -> dict:
    """
    Parse a FishLWSummary.php page into a dict.
    We read paired labels/values laid out as text nodes.
    """
    resp, final = get_with_mirror(session, detail_url)
    time.sleep(delay)
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    out = {"SourceDetailURL": final}

    # Species name near page title
    title = soup.find(string=re.compile(r"Length-Weight Relationship for", re.I))
    if title:
        # Next italicized species name often sits nearby
        sp_tag = soup.find(["i", "em"])
        if sp_tag:
            out["Species"] = norm(sp_tag.get_text(" ", strip=True))

    # The page is rendered as label blocks followed by values.
    # We'll walk all text pieces and pair them up using LABEL_MAP keys.
    texts = [norm(t) for t in soup.stripped_strings]
    i = 0
    while i < len(texts):
        t = texts[i].lower()
        if t in LABEL_MAP:
            key = LABEL_MAP[t]
            # Value is typically next text item (may be empty)
            val = ""
            if i + 1 < len(texts):
                val = texts[i + 1]
            out[key] = val
            i += 2
        else:
            i += 1

    # Some fields need light cleanup
    # n (Number of fish) -> digits only if present
    if "n" in out:
        m = re.search(r"\d+", out["n"])
        if m:
            out["n"] = int(m.group(0))

    # Length (cm): e.g., "14.0 - 53.5 FL"
    if "Length (cm)" in out:
        out["Length (cm)"] = out["Length (cm)"].replace("\u00a0", " ")

    # a / b numeric
    for k in ["a", "b", "r2"]:
        if k in out:
            s = out[k].replace(",", "")
            m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
            if m:
                try:
                    out[k] = float(m.group(0))
                except:
                    pass

    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-url", required=True)
    ap.add_argument("--out", default="assets/fishbase_lw_raw.xlsx")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--max-species", type=int, default=0, help="Process first N species only (0 = all)")
    args = ap.parse_args()

    session = build_session(args.timeout)

    print("[1/3] Reading list page and collecting species URLs…", flush=True)
    sp_map = collect_species_lw_urls(session, args.start_url, args.delay)
    if not sp_map:
        print("No species found — try the .se mirror in --start-url", file=sys.stderr)
        sys.exit(2)
    items = sorted(sp_map.items(), key=lambda kv: kv[0])
    if args.max_species > 0:
        items = items[:args.max_species]
    print(f"   Found {len(sp_map)} species; scraping {len(items)}", flush=True)

    all_rows = []
    index_rows = []

    print("[2/3] Visiting each species page and parsing all study details…", flush=True)
    for idx, (sp_id, sp_url) in enumerate(items, start=1):
        try:
            # collect detail links
            detail_links = collect_detail_links_from_species(session, sp_url, args.delay)
            if not detail_links:
                print(f"  [{idx}] ID={sp_id}: 0 studies")
                continue

            species_name_guess = None
            species_rows = []
            for durl in detail_links:
                row = parse_detail(session, durl, args.delay)
                # Species name may be present here
                if not species_name_guess and "Species" in row:
                    species_name_guess = row["Species"]
                row["SpeciesID"] = sp_id
                row["SpeciesLWURL"] = sp_url
                species_rows.append(row)

            # Fill species name if missing in some rows
            for r in species_rows:
                if "Species" not in r and species_name_guess:
                    r["Species"] = species_name_guess

            all_rows.extend(species_rows)
            index_rows.append({"SpeciesID": sp_id, "Species": species_name_guess or f"ID {sp_id}", "RowCount": len(species_rows)})

            print(f"  [{idx}] {species_name_guess or f'ID {sp_id}'}: {len(species_rows)} rows")
        except Exception as e:
            print(f"  [{idx}] ERROR for ID={sp_id}: {e}", file=sys.stderr)
            continue

    if not all_rows:
        print("Parsed 0 rows — site layout may have changed.", file=sys.stderr)
        sys.exit(3)

    # Tidy DataFrame (ensure consistent columns)
    df = pd.DataFrame(all_rows)
    # Canonical column order
    preferred = [
        "Species", "SpeciesID",
        "a", "b", "r2", "Doubtful?", "Sex", "n",
        "Length (cm)", "Territory", "Locality", "Method",
        "MainRef", "DataRef",
        "SpeciesLWURL", "SourceDetailURL",
    ]
    # put known columns first
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df.loc[:, cols]

    index_df = pd.DataFrame(index_rows).sort_values(["RowCount", "Species"], ascending=[False, True])

    print(f"[3/3] Writing Excel -> {args.out}", flush=True)
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="All_LW")
        index_df.to_excel(w, index=False, sheet_name="Index")

    print(f"Done. Wrote {len(df)} rows across {index_df['Species'].nunique()} species to {args.out}")

if __name__ == "__main__":
    main()
