import os
import re
import json
import time
import requests
import pandas as pd
import yfinance as yf
import chromadb

from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw_docs")
COMPANY_CSV = os.path.join(DATA_DIR, "companies.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def get_sec_headers() -> dict:
    """
    SEC requires a proper User-Agent for automated requests
    format: <AppName> <Name> <Email>
    """
    name = os.getenv("SEC_NAME", "Bright")
    email = os.getenv("SEC_EMAIL", "shengwh4@uci.edu")

    user_agent = f"NasdaqRAGBot {name} {email}"

    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def load_nasdaq_list() -> pd.DataFrame:
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&exchange=nasdaq&download=true"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    rows = data["data"]["rows"]
    df = pd.DataFrame(rows)

    df = df.rename(columns={"symbol": "ticker", "name": "company"})
    df = df[["ticker", "company"]].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["company"] = df["company"].astype(str).str.strip()

    df = df[df["ticker"] != ""].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df


def load_sec_ticker_cik_map() -> pd.DataFrame:
    """
    Load SEC ticker -> CIK mapping.

    Primary source:
    https://www.sec.gov/files/company_tickers.json

    Fallback:
    https://www.sec.gov/include/ticker.txt
    """
    headers = get_sec_headers()

    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()

        payload = r.json()
        rows = []

        for _, item in payload.items():
            ticker = str(item.get("ticker", "")).upper().strip()
            company = str(item.get("title", "")).strip()
            cik = item.get("cik_str")

            if ticker and cik:
                rows.append(
                    {
                        "ticker": ticker,
                        "company_sec": company,
                        "cik": int(cik),
                    }
                )

        df = pd.DataFrame(rows)
        print(f"SEC ticker map loaded: {len(df)} rows")
        return df

    except Exception as e:
        print(f"Primary SEC mapping failed: {e}")

    try:
        url = "https://www.sec.gov/include/ticker.txt"
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()

        rows = []
        for line in r.text.splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue

            ticker = parts[0].upper()
            cik = int(parts[1])

            rows.append(
                {
                    "ticker": ticker,
                    "company_sec": "",
                    "cik": cik,
                }
            )

        df = pd.DataFrame(rows)
        print(f"Fallback SEC ticker map loaded: {len(df)} rows")
        return df

    except Exception as e:
        print(f"SEC fallback mapping failed: {e}")

    print("SEC mapping unavailable")
    return pd.DataFrame(columns=["ticker", "company_sec", "cik"])


def build_company_master() -> pd.DataFrame:
    """
    建立公司主檔：
    - 主體仍以 Nasdaq screener 為準
    - 若 SEC ticker map 成功，則合併 CIK
    - 若 SEC ticker map 失敗，仍保留 Nasdaq master，不讓流程中斷
    """
    nasdaq_df = load_nasdaq_list()
    sec_map_df = load_sec_ticker_cik_map()

    if sec_map_df.empty:
        merged = nasdaq_df.copy()
        merged["cik"] = pd.NA
    else:
        merged = nasdaq_df.merge(sec_map_df[["ticker", "cik"]], on="ticker", how="left")

    merged.to_csv(COMPANY_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved company master to {COMPANY_CSV} | rows={len(merged)}")

    return merged


def paragraph_chunk(text: str, max_chars: int = 900, overlap_chars: int = 120) -> list[str]:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = f"{current}\n\n{p}".strip()
        else:
            if current:
                chunks.append(current)

            if len(p) <= max_chars:
                current = p
            else:
                start = 0
                while start < len(p):
                    end = start + max_chars
                    piece = p[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    start = max(end - overlap_chars, start + 1)
                current = ""

    if current:
        chunks.append(current)

    return chunks


def save_raw_doc(ticker: str, doc_type: str, index_num: int, title: str, text: str) -> None:
    safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title)[:60]
    path = os.path.join(RAW_DIR, f"{ticker}_{doc_type}_{index_num}_{safe_title}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def latest_fact(companyfacts: dict, taxonomy: str, concept: str, unit_priority: list[str] | None = None) -> str:
    unit_priority = unit_priority or ["USD", "USD/shares", "shares"]
    try:
        units = companyfacts["facts"][taxonomy][concept]["units"]
    except Exception:
        return ""

    best_items = None
    best_unit = None
    for u in unit_priority:
        if u in units:
            best_items = units[u]
            best_unit = u
            break

    if not best_items:
        for u, items in units.items():
            best_items = items
            best_unit = u
            break

    if not best_items:
        return ""

    items = sorted(
        best_items,
        key=lambda x: (
            safe_text(x.get("fy")),
            safe_text(x.get("fp")),
            safe_text(x.get("end")),
            safe_text(x.get("filed")),
        ),
        reverse=True,
    )

    top = items[0]
    return (
        f"{concept}: {safe_text(top.get('val'))} | "
        f"unit={best_unit} | filed={safe_text(top.get('filed'))} | "
        f"period_end={safe_text(top.get('end'))}"
    )


def build_yfinance_docs(ticker: str, company: str) -> list[dict]:
    docs = []
    tk = yf.Ticker(ticker)

    try:
        info = tk.info or {}
    except Exception:
        info = {}

    profile_lines = [
        f"Company: {company}",
        f"Ticker: {ticker}",
        f"Sector: {safe_text(info.get('sector'))}",
        f"Industry: {safe_text(info.get('industry'))}",
        f"Country: {safe_text(info.get('country'))}",
        f"City: {safe_text(info.get('city'))}",
        f"Website: {safe_text(info.get('website'))}",
        f"Employees: {safe_text(info.get('fullTimeEmployees'))}",
        f"Business Summary: {safe_text(info.get('longBusinessSummary'))}",
    ]
    profile_text = "\n".join([x for x in profile_lines if x.strip()])
    if len(profile_text) > 80:
        docs.append(
            {
                "doc_type": "yf_company_profile",
                "title": f"{ticker} Yahoo Finance Company Profile",
                "text": profile_text,
            }
        )

    financial_lines = [
        f"Company: {company}",
        f"Ticker: {ticker}",
        f"Market Cap: {safe_text(info.get('marketCap'))}",
        f"Enterprise Value: {safe_text(info.get('enterpriseValue'))}",
        f"Total Revenue: {safe_text(info.get('totalRevenue'))}",
        f"Revenue Per Share: {safe_text(info.get('revenuePerShare'))}",
        f"EBITDA: {safe_text(info.get('ebitda'))}",
        f"Gross Margins: {safe_text(info.get('grossMargins'))}",
        f"Operating Margins: {safe_text(info.get('operatingMargins'))}",
        f"Profit Margins: {safe_text(info.get('profitMargins'))}",
        f"Free Cashflow: {safe_text(info.get('freeCashflow'))}",
        f"Operating Cashflow: {safe_text(info.get('operatingCashflow'))}",
        f"Return On Assets: {safe_text(info.get('returnOnAssets'))}",
        f"Return On Equity: {safe_text(info.get('returnOnEquity'))}",
        f"Total Cash: {safe_text(info.get('totalCash'))}",
        f"Total Debt: {safe_text(info.get('totalDebt'))}",
        f"Debt To Equity: {safe_text(info.get('debtToEquity'))}",
        f"Current Ratio: {safe_text(info.get('currentRatio'))}",
        f"Quick Ratio: {safe_text(info.get('quickRatio'))}",
        f"Trailing PE: {safe_text(info.get('trailingPE'))}",
        f"Forward PE: {safe_text(info.get('forwardPE'))}",
        f"Price To Book: {safe_text(info.get('priceToBook'))}",
        f"Beta: {safe_text(info.get('beta'))}",
        f"52 Week High: {safe_text(info.get('fiftyTwoWeekHigh'))}",
        f"52 Week Low: {safe_text(info.get('fiftyTwoWeekLow'))}",
    ]
    financial_text = "\n".join([x for x in financial_lines if x.strip()])
    if len(financial_text) > 80:
        docs.append(
            {
                "doc_type": "yf_financial_snapshot",
                "title": f"{ticker} Yahoo Finance Financial Snapshot",
                "text": financial_text,
            }
        )

    try:
        news_items = tk.news or []
    except Exception:
        news_items = []

    if news_items:
        for idx, item in enumerate(news_items[:10], start=1):
            title = safe_text(item.get("title"))
            publisher = safe_text(item.get("publisher"))
            summary = safe_text(item.get("summary"))
            link = safe_text(item.get("link"))
            provider_time = safe_text(item.get("providerPublishTime"))

            body = "\n".join(
                [
                    f"Company: {company}",
                    f"Ticker: {ticker}",
                    f"Title: {title}",
                    f"Publisher: {publisher}",
                    f"Publish Time: {provider_time}",
                    f"Summary: {summary}",
                    f"Link: {link}",
                ]
            )

            if len(body) > 80:
                docs.append(
                    {
                        "doc_type": "yf_news",
                        "title": f"{ticker} Yahoo News {idx}: {title[:80]}",
                        "text": body,
                    }
                )

    return docs


def build_sec_docs(ticker: str, company: str, cik: int | float | None) -> list[dict]:
    docs = []

    if pd.isna(cik):
        return docs

    cik_int = int(cik)
    cik10 = str(cik_int).zfill(10)
    headers = get_sec_headers()

    time.sleep(0.2)

    try:
        sub_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
        r = requests.get(sub_url, headers=headers, timeout=60)

        if r.status_code != 200:
            print(f"SEC submissions skipped {ticker}")
            return docs

        _ = r.json()

    except Exception as e:
        print(f"SEC submissions failed for {ticker}: {e}")
        return docs

    try:
        cf_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
        time.sleep(0.2)

        r = requests.get(cf_url, headers=headers, timeout=60)

        if r.status_code != 200:
            print(f"SEC companyfacts skipped {ticker} status={r.status_code}")
            return docs

        companyfacts = r.json()

        if "facts" not in companyfacts:
            print(f"No companyfacts data for {ticker}")
            return docs

        facts_of_interest = [
            ("us-gaap", "Revenues"),
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "Assets"),
            ("us-gaap", "Liabilities"),
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ]

        lines = [
            f"Company: {company}",
            f"Ticker: {ticker}",
            f"CIK: {cik10}",
            "Latest SEC XBRL Facts:",
        ]

        for taxonomy, concept in facts_of_interest:
            try:
                item = latest_fact(companyfacts, taxonomy, concept)
            except Exception:
                item = ""

            if item:
                lines.append(item)

        facts_text = "\n".join(lines)
        if len(facts_text) > 120:
            docs.append(
                {
                    "doc_type": "sec_companyfacts",
                    "title": f"{ticker} SEC Company Facts Snapshot",
                    "text": facts_text,
                }
            )

    except Exception as e:
        print(f"SEC companyfacts failed for {ticker}: {e}")

    return docs


def reset_collection(client: chromadb.PersistentClient, collection_name: str) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


def build_index(
    chroma_dir: str = "chroma_db",
    collection_name: str = "nasdaq_docs",
    limit: int | None = 30,
) -> None:
    """
    Streamlit / GitHub deploy friendly version:
    - 支援 app.py 傳入 chroma_dir / collection_name / limit
    - 初次部署建議 limit=20~30，避免 Streamlit cold start 太久
    """
    company_master = build_company_master()
    print(f"Saved company master to {COMPANY_CSV}")
    print(f"Nasdaq companies loaded: {len(company_master)}")

    client = chromadb.PersistentClient(path=chroma_dir)
    reset_collection(client, collection_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    work_df = company_master.copy()
    if limit is not None:
        work_df = work_df.head(limit)

    total_docs = 0
    total_chunks = 0

    for i, row in work_df.iterrows():
        ticker = row["ticker"]
        company = row["company"]
        cik = row.get("cik", None)

        print(f"[{i+1}/{len(work_df)}] Building docs for {ticker} - {company}")

        docs = []

        try:
            docs.extend(build_yfinance_docs(ticker, company))
        except Exception as e:
            print(f"YF doc build failed for {ticker}: {e}")

        try:
            docs.extend(build_sec_docs(ticker, company, cik))
        except Exception as e:
            print(f"SEC doc build failed for {ticker}: {e}")

        if not docs:
            continue

        for doc_idx, doc in enumerate(docs):
            text = doc["text"]
            title = doc["title"]
            doc_type = doc["doc_type"]

            if not text or len(text) < 80:
                continue

            save_raw_doc(ticker, doc_type, doc_idx, title, text)

            chunks = paragraph_chunk(text)
            if not chunks:
                continue

            total_docs += 1

            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{ticker}_{doc_type}_{doc_idx}_{chunk_idx}"
                metadata = {
                    "ticker": ticker,
                    "company": company,
                    "source": "sec" if doc_type.startswith("sec_") else "yfinance",
                    "doc_type": doc_type,
                    "title": title,
                    "chunk_index": chunk_idx,
                }

                collection.add(
                    ids=[chunk_id],
                    documents=[chunk],
                    metadatas=[metadata],
                )
                total_chunks += 1

        time.sleep(0.2)

    print(f"Collection count: {collection.count()}")
    print(f"Total documents indexed: {total_docs}")
    print(f"Total chunks indexed: {total_chunks}")


if __name__ == "__main__":
    # 初次本機測試建議 20 或 30
    build_index(limit=30)
