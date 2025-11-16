Trade Watch Backend (FastAPI)
-----------------------------

1. Copy .env.sample to .env and edit paths. Most defaults are fine.
2. Place full EQUITY_L.csv into backend/data/EQUITY_L.csv
3. Build and run:
   - Docker:
       docker build -t trade-watch-backend .
       docker run -d -p 8000:8000 --name trade-watch-backend trade-watch-backend
   - Or locally:
       python -m venv venv
       source venv/bin/activate   # or venv\Scripts\activate on Windows
       pip install -r requirements.txt
       uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

4. Open http://localhost:8000/ to see API (the frontend must be placed in a static server or served by the backend).
5. Scheduler: the app will schedule daily scan using APScheduler at the time specified by SCHEDULE_UTC in .env (HH:MM). For 9:20 AM IST use 03:50 (UTC).

Security notes:
- PIN is stored as SHA-256 hash in the file specified by PIN_HASH_FILE.
- WebAuthn credentials are stored as raw blobs; for production you should integrate a proper FIDO server to validate signatures.
