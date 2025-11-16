from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os, json
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ----- env -----
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
EQUITY_CSV = os.getenv("EQUITY_CSV", str(ROOT / "/data/EQUITY_L.csv"))
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "200"))
OUTPUT_FILE = ROOT / "data" / "daily_watchlist.csv"
CREDENTIALS_FILE = Path(os.getenv("CREDENTIALS_FILE", str(ROOT / "data/webauthn_credentials.json")))

# Ensure data dir
Path(ROOT / "data").mkdir(parents=True, exist_ok=True)
if not CREDENTIALS_FILE.exists():
    CREDENTIALS_FILE.write_text("[]", encoding="utf-8")

app = FastAPI(title="Trade Watch API")

# ----- simple scanner -----
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_LO = 30
RSI_HI = 70

def load_tickers():
    csv_path = Path(EQUITY_CSV)
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "SERIES" in df.columns and "SYMBOL" in df.columns:
        df = df[df["SERIES"] == "EQ"]
        return [s + ".NS" for s in df["SYMBOL"].astype(str).tolist()]
    else:
        syms = df.iloc[:,0].astype(str).tolist()
        return [s + ".NS" for s in syms]

def fetch_hist(ticker):
    try:
        df = yf.download(ticker, period="90d", interval="1d", progress=False)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print("yfinance err", e)
        return None

def compute(df):
    df = df.copy()
    df["ema_fast"] = EMAIndicator(df["Close"], EMA_FAST).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["Close"], EMA_SLOW).ema_indicator()
    df["rsi"] = RSIIndicator(df["Close"], RSI_PERIOD).rsi()
    df["mom5"] = df["Close"].pct_change(5)
    return df

def score_last(df):
    last = df.iloc[-1]
    score = 0.0
    if last["ema_fast"] > last["ema_slow"]:
        score += 1.5
    else:
        score -= 0.5
    if RSI_LO < last["rsi"] < RSI_HI:
        score += 1.0
    else:
        score -= 0.5
    score += float(np.clip(last["mom5"] * 10, -1, 1))
    return score

def run_scan(limit=SCAN_LIMIT):
    tickers = load_tickers()
    rows = []
    count = 0
    for t in tickers:
        if limit and count >= limit:
            break
        df = fetch_hist(t)
        if df is None or len(df) < EMA_SLOW + 5:
            continue
        try:
            df = compute(df)
            score = score_last(df)
            rows.append({
                "ticker": t,
                "close": float(df["Close"].iloc[-1]),
                "score": float(round(score,3)),
                "rsi": float(round(df["rsi"].iloc[-1],1)),
                "ema_fast": float(round(df["ema_fast"].iloc[-1],2)),
                "ema_slow": float(round(df["ema_slow"].iloc[-1],2)),
                "mom5": float(round(df["mom5"].iloc[-1],4)),
            })
            count += 1
        except Exception as e:
            print("compute error", e)
    df_out = pd.DataFrame(rows)
    if df_out.empty:
        df_out = pd.DataFrame(columns=[
            "ticker", "close", "score", "rsi", "ema_fast", "ema_slow", "mom5"
        ])
    df_out.to_csv(OUTPUT_FILE, index=False)
    df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)
    return df_out

# ----- API endpoints -----
@app.get("/api/scan")
async def api_scan():
    df = run_scan()
    top = df.head(50).to_dict(orient="records")
    return {"ok": True, "generated_at": datetime.utcnow().isoformat()+"Z", "top": top}

@app.get("/api/today")
async def api_today():
    if OUTPUT_FILE.exists():
        df = pd.read_csv(OUTPUT_FILE)
        return {"ok": True, "top": df.head(50).to_dict(orient="records")}
    else:
        return {"ok": False, "message": "No scan available"}

@app.get("/api/download")
async def api_download():
    if OUTPUT_FILE.exists():
        return FileResponse(str(OUTPUT_FILE), media_type="text/csv", filename="daily_watchlist.csv")
    else:
        raise HTTPException(status_code=404, detail="Not found")

# ----- Scheduler -----
scheduler = AsyncIOScheduler()
async def scheduled_job():
    print("Running scheduled scan at", datetime.utcnow().isoformat())
    run_scan()
scheduler.add_job(scheduled_job, 'cron', hour=int(os.getenv("SCHEDULE_UTC","03:50").split(":")[0]),
                  minute=int(os.getenv("SCHEDULE_UTC","03:50").split(":")[1]))

@app.get("/")
def home():
    return {"status": "ok", "message": "Trade Watch Backend is running"}

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    if not OUTPUT_FILE.exists():
        run_scan(limit=50)
