# test_watermark_ingest.py
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL = "https://data.boston.gov/api/3/action/datastore_search_sql"
RESOURCE_ID = "b973d8cb-eeb2-4e7e-99da-c92938efc9c0"
BATCH_SIZE = 1000
LOOKBACK_DAYS = 3

# For now store watermark locally (swap for GCS later)
WATERMARK_FILE = Path("watermark_crime.json")


# ─────────────────────────────────────────────
# WATERMARK FUNCTIONS
# ─────────────────────────────────────────────


def read_watermark() -> datetime | None:
    """
    Read the last date we successfully pulled data up to.
    Returns None if this is the very first run.
    """
    if not WATERMARK_FILE.exists():
        print("No watermark file found → this is the first run")
        return None

    with open(WATERMARK_FILE) as f:
        data = json.load(f)

    watermark = datetime.fromisoformat(data["last_successful_date"])
    print(f"Watermark found: {watermark.date()}")
    return watermark


def write_watermark(date: datetime):
    """
    Save the date we successfully pulled data up to.
    ONLY called after data is safely written — never before.
    """
    with open(WATERMARK_FILE, "w") as f:
        json.dump(
            {
                "last_successful_date": date.isoformat(),
                "written_at": datetime.now(UTC).isoformat(),
            },
            f,
            indent=2,
        )

    print(f"Watermark saved → {date.date()}")


# ─────────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────────


def fetch_page(since: str, until: str, offset: int) -> list[dict]:
    """
    Fetch one page of crime records between two dates.
    since/until format: "2024-01-15"
    """
    sql = (
        f'SELECT * FROM "{RESOURCE_ID}" '
        f"WHERE \"OCCURRED_ON_DATE\" >= '{since} 00:00' "
        f"AND \"OCCURRED_ON_DATE\" <= '{until} 23:59' "
        f'ORDER BY "OCCURRED_ON_DATE" ASC '
        f"LIMIT {BATCH_SIZE} OFFSET {offset}"
    )

    response = requests.get(BASE_URL, params={"sql": sql}, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

    data = response.json()

    if not data["success"]:
        raise RuntimeError(f"API error: {data['error']}")

    return data["result"]["records"]


def fetch_all(since: str, until: str) -> pd.DataFrame:
    """
    Pull ALL records between since and until.
    Paginates automatically.
    """
    all_records = []
    offset = 0

    print(f"Fetching crimes from {since} to {until}...")

    while True:
        print(f"  Requesting offset {offset}...")
        page = fetch_page(since, until, offset)

        if not page:
            print("  No more records.")
            break

        all_records.extend(page)
        offset += len(page)
        print(f"  Got {len(page)} records. Total so far: {offset}")

        if len(page) < BATCH_SIZE:
            print("  Last page reached.")
            break

        time.sleep(0.5)  # be polite to the API

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df = df.drop(columns=["_id", "_full_text"], errors="ignore")
    return df


# ─────────────────────────────────────────────
# MAIN INGEST WITH WATERMARK
# ─────────────────────────────────────────────


def ingest(execution_date: str) -> pd.DataFrame:
    """
    Main ingest function.
    Uses watermark to only pull NEW data each run.

    First run:  pulls last LOOKBACK_DAYS days
    Later runs: pulls from (watermark - LOOKBACK_DAYS) to execution_date
    """
    print("\n" + "=" * 50)
    print(f"INGEST RUN: {execution_date}")
    print("=" * 50)

    exec_dt = datetime.strptime(execution_date, "%Y-%m-%d")

    # ── Step 1: Read watermark ──────────────────
    watermark = read_watermark()

    if watermark:
        # We ran before — start from watermark minus lookback
        # The lookback overlap catches any late-arriving records
        # Example: BPD sometimes files incident reports a day or two late
        since_dt = watermark - timedelta(days=LOOKBACK_DAYS)
    else:
        # First ever run — just go back LOOKBACK_DAYS from today
        since_dt = exec_dt - timedelta(days=LOOKBACK_DAYS)

    since = since_dt.strftime("%Y-%m-%d")
    until = exec_dt.strftime("%Y-%m-%d")

    print(f"Date range: {since} → {until}")

    # ── Step 2: Fetch data ──────────────────────
    try:
        df = fetch_all(since, until)
    except Exception as e:
        # Fetch failed — do NOT update watermark
        # Next run will retry from same starting point
        print(f"FETCH FAILED: {e}")
        print("Watermark NOT updated — next run will retry same range")
        raise

    if df.empty:
        print("No records found in range")
        # Still update watermark so we move forward
        write_watermark(exec_dt)
        return df

    print(f"\nFetch complete: {len(df)} records")

    # ── Step 3: Update watermark ────────────────
    # ONLY after data is safely in hand
    write_watermark(exec_dt)

    return df


# ─────────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Simulate running the pipeline today
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    df = ingest(today)

    if not df.empty:
        print("\n── Sample Data ──────────────────")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df[["INCIDENT_NUMBER", "OCCURRED_ON_DATE", "OFFENSE_CODE_GROUP", "DISTRICT"]].head(5))

    # ── Simulate running again tomorrow ─────────
    # This shows how watermark prevents re-fetching old data
    print("\n\nSimulating tomorrow's run...")
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime("%Y-%m-%d")
    df2 = ingest(tomorrow)
    print(f"Second run got: {len(df2)} records")
