from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "datasets" / "output"
TEXT_PATH = OUTPUT_DIR / "pmc_biomedical_corpus.txt"
JSONL_PATH = OUTPUT_DIR / "pmc_biomedical_corpus.jsonl"
MANIFEST_PATH = ROOT / "datasets" / "manifest.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def estimate_tokens(text: str) -> int:
    return max(1, round(len(re.findall(r"\w+", text)) * 1.32))


def request_json(endpoint: str, params: dict[str, str | int]) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{EUTILS}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def request_xml(endpoint: str, params: dict[str, str | int]) -> ET.Element:
    query = urllib.parse.urlencode(params)
    url = f"{EUTILS}/{endpoint}?{query}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return ET.fromstring(response.read())


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def node_text(node: ET.Element) -> str:
    return " ".join(part.strip() for part in node.itertext() if part and part.strip())


def extract_article(article: ET.Element, fallback_id: str) -> dict[str, str]:
    title = ""
    pmcid = fallback_id
    license_text = "PMC Open Access"
    sections: list[str] = []

    for node in article.iter():
        tag = strip_namespace(node.tag)
        if tag == "article-id" and node.attrib.get("pub-id-type") == "pmc":
            pmcid = node_text(node) or fallback_id
        elif tag == "article-title" and not title:
            title = node_text(node)
        elif tag == "license-p":
            license_text = node_text(node)[:500] or license_text
        elif tag in {"abstract", "body"}:
            for child in node.iter():
                if strip_namespace(child.tag) == "p":
                    text = node_text(child)
                    if len(text.split()) >= 20:
                        sections.append(text)

    text = "\n\n".join(dict.fromkeys(sections))
    return {
        "pmcid": pmcid,
        "title": html.unescape(title or f"PMC article {fallback_id}"),
        "license": html.unescape(license_text),
        "text": html.unescape(text),
    }


def search_ids(term: str, retmax: int, email: str | None) -> list[str]:
    params: dict[str, str | int] = {
        "db": "pmc",
        "term": f"({term}) AND open access[filter]",
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    }
    if email:
        params["email"] = email
    data = request_json("esearch.fcgi", params)
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_articles(ids: list[str], email: str | None) -> list[dict[str, str]]:
    if not ids:
        return []
    params: dict[str, str | int] = {
        "db": "pmc",
        "id": ",".join(ids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    root = request_xml("efetch.fcgi", params)
    articles = [node for node in root.iter() if strip_namespace(node.tag) == "article"]
    return [extract_article(article, ids[index] if index < len(ids) else str(index)) for index, article in enumerate(articles)]


def build_dataset(target_tokens: int, retmax: int, batch_size: int, term: str, email: str | None) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = search_ids(term, retmax=retmax, email=email)
    total_tokens = 0
    docs_written = 0
    licenses: dict[str, int] = {}

    with JSONL_PATH.open("w", encoding="utf-8") as jsonl, TEXT_PATH.open("w", encoding="utf-8") as text_out:
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            for article in fetch_articles(batch, email=email):
                text = article["text"].strip()
                if len(text.split()) < 120:
                    continue
                tokens = estimate_tokens(text)
                row = {
                    "id": f"PMC{article['pmcid']}",
                    "title": article["title"],
                    "source": "PubMed Central Open Access",
                    "license": article["license"],
                    "token_count": tokens,
                    "text": text,
                }
                jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                text_out.write(f"\n\n# {row['id']} - {row['title']}\n\n{text}\n")
                total_tokens += tokens
                docs_written += 1
                licenses[article["license"][:120]] = licenses.get(article["license"][:120], 0) + 1
                if total_tokens >= target_tokens:
                    break
            if total_tokens >= target_tokens:
                break
            time.sleep(0.35)

    manifest = {
        "source": "PubMed Central Open Access via NCBI E-utilities",
        "query": f"({term}) AND open access[filter]",
        "target_token_count": target_tokens,
        "token_count": total_tokens,
        "document_count": docs_written,
        "target_met": total_tokens >= target_tokens,
        "jsonl_path": str(JSONL_PATH.relative_to(ROOT)),
        "text_path": str(TEXT_PATH.relative_to(ROOT)),
        "license_note": "Records are fetched from the PMC Open Access subset; per-article license text is preserved in the JSONL rows.",
        "license_samples": licenses,
        "created_by": "datasets/build_open_access_pmc_dataset.py",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Round 1 2M-token biomedical dataset from PMC Open Access.")
    parser.add_argument("--target-tokens", type=int, default=2_000_000)
    parser.add_argument("--retmax", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument(
        "--term",
        default="Alzheimer disease inflammation protein drug target cytokine neurodegeneration",
    )
    parser.add_argument("--email", default=None, help="Optional email for NCBI E-utilities courtesy metadata.")
    args = parser.parse_args()

    manifest = build_dataset(
        target_tokens=args.target_tokens,
        retmax=args.retmax,
        batch_size=args.batch_size,
        term=args.term,
        email=args.email,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
