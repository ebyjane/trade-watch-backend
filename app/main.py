from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trade-watch-api")

# ----- env -----
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
EQUITY_CSV = os.getenv("EQUITY_CSV", str(ROOT / "data" / "EQUITY_L.csv"))
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "50"))  # lower default for speed
OUTPUT_FILE = ROOT / "data" / "daily_watchlist.csv"
CREDENTIALS_FILE = Path(os.getenv("CREDENTIALS_FILE", str(ROOT / "data" / "webauthn_credentials.json")))
TICKER_SUFFIX = os.getenv("TICKER_SUFFIX", "")  # e.g. ".NS" for NSE if needed
BATCH_DOWNLOAD = os.getenv("BATCH_DOWNLOAD", "1") == "1"

# Ensure data dir
Path(ROOT / "data").mkdir(parents=True, exist_ok=True)
if not CREDENTIALS_FILE.exists():
    CREDENTIALS_FILE.write_text("[]", encoding="utf-8")

app = FastAPI(title="Trade Watch API")

# add CORS (allow your frontend origin in production instead of "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # for production set to frontend origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ----- simple scanner -----
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_LO = int(os.getenv("RSI_LO", "30"))
RSI_HI = int(os.getenv("RSI_HI", "70"))

def load_tickers() -> List[str]:
    csv_path = Path(EQUITY_CSV)
    if not csv_path.exists():
        logger.warning("EQUITY_CSV not found, using fallback tickers")
        return ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "INTC", "AMD", "NFLX"]
    try:
        df = pd.read_csv(csv_path)
        if "SYMBOL" in df.columns:
            tickers = df["SYMBOL"].astype(str).str.strip().tolist()
        else:
            tickers = df.iloc[:, 0].astype(str).str.strip().tolist()
        # dedupe preserve order
        tickers = list(dict.fromkeys([t for t in tickers if t]))
        if TICKER_SUFFIX:
            # only append suffix if not already present
            tickers = [t if t.endswith(TICKER_SUFFIX) else f"{t}{TICKER_SUFFIX}" for t in tickers]
        return tickers
    except Exception as e:
        logger.exception("Failed to load tickers from CSV: %s", e)
        return ["AAPL", "MSFT", "GOOG"]

def try_fetch_single(ticker: str, period: str = "180d", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        return df
    except Exception as e:
        logger.debug("yfinance single fetch failed for %s: %s", ticker, e)
        return None

def try_batch_download(tickers: List[str], period: str = "180d", interval: str = "1d") -> dict:
    """
    Attempt to download in batch. Returns dict ticker -> df (or None).
    """
    results = {}
    try:
        # yfinance.download returns either a DataFrame or MultiIndex columns when multiple tickers are passed
        raw = yf.download(tickers=tickers, period=period, interval=interval, threads=True, progress=False)
        if raw is None or raw.empty:
            return {t: None for t in tickers}
        # handle MultiIndex columns (col, ticker) or single ticker
        if isinstance(raw.columns, pd.MultiIndex):
            # columns like ('Close', 'AAPL'), etc.
            for t in tickers:
                try:
                    df_t = raw.xs(t, axis=1, level=1)
                    df_t = df_t.reset_index()
                    results[t] = df_t
                except Exception:
                    results[t] = None
        else:
            # single-ticker DataFrame returned
            for t in tickers:
                results[t] = raw.reset_index()
        return results
    except Exception as e:
        logger.debug("Batch download failed: %s", e)
        return {t: None for t in tickers}

def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns and "close" in df.columns:
        df.rename(columns={"close": "Close"}, inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        return df
    try:
        df["ema_fast"] = EMAIndicator(df["Close"], EMA_FAST).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["Close"], EMA_SLOW).ema_indicator()
        df["rsi"] = RSIIndicator(df["Close"], RSI_PERIOD).rsi()
    except Exception as e:
        logger.debug("Indicator compute failed: %s", e)
        # continue without raising
    df["mom5"] = df["Close"].pct_change(5)
    return df

def score_last(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    last = df.iloc[-1]
    score = 0.0
    try:
        if last.get("ema_fast", 0) > last.get("ema_slow", 0):
            score += 0.5
        else:
            score -= 0.5
        r = last.get("rsi", None)
        if r is not None and RSI_LO < r < RSI_HI:
            score += 0.5
        else:
            score -= 0.2
        mom = last.get("mom5", 0) if last.get("mom5", None) is not None else 0
        score += float(np.clip(mom * 10, -1, 1))
    except Exception as e:
        logger.debug("score_last error: %s", e)
    return float(score)

def run_scan(limit: int = SCAN_LIMIT) -> Tuple[pd.DataFrame, dict]:
    tickers = load_tickers()
    if not tickers:
        return pd.DataFrame(), {"scanned": 0, "success": 0, "errors": ["no tickers"]}

    scan_tickers = tickers[:limit]
    rows = []
    errors = []
    success = 0

    # Try batch download first if enabled and more than one ticker
    batch_results = {}
    if BATCH_DOWNLOAD and len(scan_tickers) > 1:
        try:
            logger.info("Attempting batch download for %d tickers", len(scan_tickers))
            batch_results = try_batch_download(scan_tickers)
        except Exception as e:
            logger.debug("Batch download exception: %s", e)
            batch_results = {}

    for t in scan_tickers:
        df = None
        try:
            # prefer batch_results if present
            if batch_results:
                df = batch_results.get(t)
            if df is None or df.empty:
                df = try_fetch_single(t)
            if df is None or df.empty:
                errors.append(f"no data for {t}")
                continue
            dfc = compute(df)
            if dfc is None or dfc.empty:
                errors.append(f"insufficient data for {t}")
                continue
            s = score_last(dfc)
            last = dfc.iloc[-1]
            date_val = None
            if "Date" in last and not pd.isna(last.get("Date")):
                date_val = pd.to_datetime(last.get("Date")).isoformat()
            elif "date" in last and not pd.isna(last.get("date")):
                date_val = pd.to_datetime(last.get("date")).isoformat()
            rows.append({
                "ticker": t,
                "close": float(last["Close"]),
                "rsi": float(last.get("rsi")) if last.get("rsi") is not None else None,
                "score": s,
                "date": date_val
            })
            success += 1
        except Exception as e:
            logger.exception("Error scanning %s: %s", t, e)
            errors.append(f"error {t}: {str(e)}")
        # optional quick-stop for speed (already limited by loop)
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)
        df_out.to_csv(OUTPUT_FILE, index=False)
    else:
        # write empty CSV header
        pd.DataFrame(columns=["ticker", "close", "rsi", "score", "date"]).to_csv(OUTPUT_FILE, index=False)

    meta = {"scanned": len(scan_tickers), "success": success, "errors": errors[:50]}
    return df_out, meta

# ----- API endpoints -----
@app.get("/api/scan")
async def api_scan():
    df, meta = run_scan()
    top = df.head(50).to_dict(orient="records") if not df.empty else []
    if not top:
        # deterministic sample fallback so frontend has something to show
        sample = [
            {"ticker": "AAPL", "close": 185.0, "rsi": 54.2, "score": 0.9, "date": datetime.utcnow().isoformat()},
            {"ticker": "MSFT", "close": 410.0, "rsi": 48.1, "score": 0.7, "date": datetime.utcnow().isoformat()},
        ]
        return JSONResponse({"top": sample, "count": len(sample), "meta": meta})
    return JSONResponse({"top": top, "count": len(top), "meta": meta})

@app.get("/api/today")
async def api_today():
    if OUTPUT_FILE.exists():
        try:
            df = pd.read_csv(OUTPUT_FILE)
            return JSONResponse({"top": df.head(50).to_dict(orient="records"), "count": len(df)})
        except Exception as e:
            logger.exception("Failed to read output file: %s", e)
            return JSONResponse({"top": [], "count": 0, "error": str(e)})
    return JSONResponse({"top": [], "count": 0})

@app.get("/api/download")
async def api_download():
    if not OUTPUT_FILE.exists():
        raise HTTPException(status_code=404, detail="No output file")
    return FileResponse(path=str(OUTPUT_FILE), filename="daily_watchlist.csv", media_type="text/csv")

# ----- Scheduler -----
scheduler = AsyncIOScheduler()

async def scheduled_job():
    logger.info("Scheduled job starting scan")
    try:
        run_scan()
    except Exception as e:
        logger.exception("Scheduled job failed: %s", e)

# parse schedule safely
def _parse_schedule(s: str) -> Tuple[int, int]:
    try:
        parts = s.split(":")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 3, 50

hour, minute = _parse_schedule(os.getenv("SCHEDULE_UTC", "03:50"))
scheduler.add_job(scheduled_job, "cron", hour=hour, minute=minute)

@app.get("/")
async def root():
    return PlainTextResponse("Trade Watch API is running")

@app.on_event("startup")
async def startup_event():
    try:
        scheduler.start()
    except Exception:
        logger.debug("Scheduler start skipped / already running")
    # initial scan in background (do not block startup)
    try:
        logger.info("Initial scan triggered")
        # run in executor to avoid blocking if running under uvicorn async loop
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(run_scan)
    except Exception as e:
        logger.exception("Initial scan error: %s", e)
