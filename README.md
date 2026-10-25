# NeuroCardio AI (frontend)

Research prototype for reviewing EEG and ECG recordings. This interface helps
clinicians organize files and helps patients learn what these signals are. It
does **not** provide a confirmed diagnosis and has no claimed clinical accuracy.

## Run locally

```bash
npm install
npm run dev
```

Then open the URL printed by Vite (usually `http://localhost:5173`).

Optional environment file (see `.env.example`):

```
VITE_API_BASE_URL=/api
```

## Current scope

- Dashboard, EEG analysis, ECG analysis, and About pages
- File type and size checks in the browser
- Waveform preview for simple CSV/TXT files
- Labeled demonstration waveforms when a clinical format cannot be parsed
- Analysis requests to FastAPI (`POST /api/eeg/analyze` and `POST /api/ecg/analyze`)

The FastAPI backend and trained models are not included yet. If analysis is not
connected, the results panel reports that honestly and does not invent findings.

## Later backend connection

Vite proxies `/api` to `http://localhost:8000`. When FastAPI is running, upload
analysis should reach:

- `POST /eeg/analyze`
- `POST /ecg/analyze`

both as `multipart/form-data` with a `file` field.
