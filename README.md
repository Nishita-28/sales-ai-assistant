# Internal AI Sales Assistant

An internal tool for the sales team. Ask it a question about our products and it answers using only
our approved documents — it never guesses, and every answer shows where it came from.

## What it does

- **Assistant** — ask any product/application question, get an answer with sources and a confidence
  level.
- **Customer Requirements** — capture what a customer needs (install type, comms, certifications,
  etc.) for handoff to production.
- **Discovery Questions** — before a sales call, generate Right-to-Win talking points and exploratory
  questions grounded in our documents.
- **Sales Aids** — generate a comparison document (us vs. a competitor) for a specific use case.
- **Admin** — manage feedback, upload new documents, view accuracy over time. Password-protected.

## Setup

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
2. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in a real `OPENAI_API_KEY`.
5. (Optional, only needed for cloud deployment) Set `DATABASE_URL` in `.env` to a Postgres connection
   string -- see the comment in `.env.example`. Without it, everything is stored in local files/SQLite,
   which is fine for local dev but doesn't survive a Streamlit Community Cloud redeploy.

## Running it

```powershell
# Build the search index (run this once, and again whenever documents change)
python -m app.rag_pipeline --reindex

# Start the app
streamlit run app/streamlit_app.py
```

## Adding documents

Put approved files in `data/approved_docs/` (PDF, DOCX, PPTX, CSV, Markdown, or text), then run the
reindex command above. Or use the upload button on the Admin page, which indexes the file
automatically.

## Persistence

Two independent stores, for two different reasons:

- **Postgres (Neon)** -- documents, Approved/Restricted Claims, deals, customer requirements,
  feedback, and generated Sales Aids, when `DATABASE_URL` is set. This is the durable source of
  truth for anything an admin or rep enters through the UI, since Streamlit Community Cloud's
  filesystem is wiped on every redeploy. Without `DATABASE_URL`, the same data lives in local
  files/SQLite under `data/` instead -- fully functional for local dev, just not durable across a
  cloud redeploy.
- **Local Chroma vector index** (`data/chroma_index/` by default) -- always local, never in
  Postgres. When `DATABASE_URL` is set, it's rebuilt automatically from Postgres the first time the
  app starts in a fresh process (cold start), so a redeploy always comes back with a correct index
  without a manual reindex step.

Don't run two Streamlit processes against the same local `data/chroma_index/` directory at the same
time (e.g. your own `streamlit run` plus a second one someone else started) -- concurrent access to
the local Chroma index can corrupt it. If that happens, stop every process touching it, delete
`data/chroma_index/`, and restart one process; it rebuilds cleanly from Postgres (or from
`data/approved_docs/` via the reindex command, if not using Postgres).

## Rules

- Only put approved, non-confidential documents in `data/approved_docs/`. No chip design, customer,
  legal, financial, or vendor documents.
- Never commit the real `.env` file or an API key.
- This is internal-only. It is not meant to be exposed publicly or connected to customers directly.

## Tests

```powershell
pytest tests/
```

Regression tests, each written against a real, confirmed failure rather than up front:

- `test_approved_claims_chunking.py` -- a newly-added Approved Claims bullet must actually surface in
  the assistant's answer after reindexing, not get diluted into an unrelated chunk.
- `test_table_structure_check.py` -- `check_table_structure` (app/document_loader.py) catches a
  misdetected table header before it silently buries real spec data underneath it.
- `test_xlsx_extraction.py` -- XLSX title/table-region detection, including the real approved
  Historical Sales workbook.

Most testing beyond this has been done by hand -- asking real questions in the app and checking the
answers, and (for admin features) exercising the actual UI end-to-end rather than only calling the
underlying functions directly. Broader automated coverage is still a work in progress.
