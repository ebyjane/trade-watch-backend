\
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os, json, hashlib, base64, asyncio
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
PIN_HASH_FILE = Path(os.getenv("PIN_HASH_FILE", str(ROOT / "data/pin_hash.txt")))
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
    # ---- SAFE DATAFRAME FIX ----
    df_out = pd.DataFrame(rows)

    # If no rows → write empty output safely
    if df_out.empty:
        print("⚠ No scan rows produced — writing empty CSV")
        df_out = pd.DataFrame(columns=[
            "ticker", "close", "score", "rsi", "ema_fast", "ema_slow", "mom5"
        ])
        df_out.to_csv(OUTPUT_FILE, index=False)
        return df_out

    # Ensure score column exists
    if "score" not in df_out.columns:
        print("⚠ Missing 'score' column — adding score=0")
        df_out["score"] = 0.0

    # Now safe to sort
    df_out = df_out.sort_values("score", ascending=False).reset_index(drop=True)

    df_out.to_csv(OUTPUT_FILE, index=False)
    return df_out


# ----- Auth: PIN and WebAuthn (best-effort minimal) -----
def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()

class PinModel(BaseModel):
    pin: str

@app.post("/api/auth/set_pin")
async def set_pin(payload: PinModel):
    pin = payload.pin.strip()
    if not pin or len(pin) < 4:
        raise HTTPException(status_code=400, detail="PIN too short")
    PIN_HASH_FILE.write_text(hash_pin(pin))
    return {"ok": True, "message": "PIN set. Keep it safe."}

@app.post("/api/auth/login_pin")
async def login_pin(payload: PinModel):
    if not PIN_HASH_FILE.exists():
        raise HTTPException(status_code=400, detail="PIN not set")
    stored = PIN_HASH_FILE.read_text().strip()
    if stored == hash_pin(payload.pin):
        return {"ok": True}
    else:
        raise HTTPException(status_code=401, detail="Invalid PIN")

# WebAuthn endpoints: we will store raw credential info client-side; full verification requires FIDO server validation.
@app.get("/api/webauthn/credentials")
async def get_credentials():
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
    except Exception:
        data = []
    return {"credentials": data}

class CredentialModel(BaseModel):
    credential: dict

@app.post("/api/webauthn/register")
async def webauthn_register(payload: CredentialModel):
    # store credential blob (client must perform registration via navigator.credentials)
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
    except Exception:
        data = []
    data.append(payload.credential)
    CREDENTIALS_FILE.write_text(json.dumps(data))
    return {"ok": True}

@app.post("/api/webauthn/authenticate")
async def webauthn_authenticate(payload: CredentialModel):
    # naive authentication: verify that credential id exists
    try:
        data = json.loads(CREDENTIALS_FILE.read_text())
    except Exception:
        data = []
    cred_id = payload.credential.get("id")
    for c in data:
        if c.get("id") == cred_id:
            return {"ok": True}
    raise HTTPException(status_code=401, detail="Credential not registered")

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
scheduler.add_job(scheduled_job, 'cron', hour=int(os.getenv("SCHEDULE_UTC","03:50").split(":")[0]), minute=int(os.getenv("SCHEDULE_UTC","03:50").split(":")[1]))

@app.on_event("startup")
async def startup_event():
    scheduler.start()
    # ensure output exists
    if not OUTPUT_FILE.exists():
        run_scan(limit=50)

