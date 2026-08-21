# Internal AI Sales Assistant (MNST)

An internal tool for the sales team, built around a single knowledge base of approved company
documents. It answers product/application questions with sources and a confidence level, and
supports the surrounding sales workflow: capturing customer requirements, running pre-call
discovery, and generating customer-facing comparison documents -- all grounded in the same
approved-document knowledge base and the same safety checks.

Everything is organized around a **deal**: a rep creates or picks a deal once, and Customer
Requirements, Discovery Questions, and Sales Aids all attach their output to it, so an admin can see
a deal's full history (requirements submitted, discovery run, sales aids generated) from one place.

## What it does

### Assistant
Ask any product or application question and get an answer generated only from approved documents,
with the source excerpts and a confidence level (High/Medium/Low) shown alongside it. Two answer
paths run underneath it:
- **Deterministic catalog lookups** -- "what products do you offer," "what's the resolution for
  PORTaHY at 5K," decoding an ordering code -- answered directly from an auto-extracted Product
  Index rather than a model guessing from retrieved text, so these can't be silently wrong or
  incomplete.
- **Retrieval-augmented generation (RAG)** -- everything else, answered by retrieving the most
  relevant approved excerpts and generating a grounded response from them.

Every answer also passes through the **claim guardrail** (see below) before being shown, and each
answer has Correct / Wrong / Unsafe feedback buttons that feed the Admin accuracy view. A rep can
also request a plain-language, customer-facing rewrite of any answer, gated by the same guardrail.

### Customer Requirements
A structured intake form for what a customer needs -- install type, number of detectors,
communication protocols, certifications, hazard zone, and more -- tied to the active deal. Product
selection surfaces that product's real selectable options (pulled from its catalogue's ordering
code) instead of a generic field list, and submissions are saved for later reference.

### Discovery Questions
Pre-call discovery for a deal: describe the customer's use case and get back documented
Right-to-Win points (each with its own evidence and exploratory questions to ask), not a product
recommendation yet. Record the customer's answers during the call, then request a product
recommendation once there's enough information -- recommendation, discovery notes, and a
qualification checklist are all saved to the deal.

### Sales Aids
Generates a short, customer-facing comparison document (MNST vs. a named or unnamed competitor) for
a specific use case -- a table of relevant topics plus a plain-prose summary suitable for pasting
into an email. Every generated Sales Aid is saved to its deal, so a rep can regenerate for a new
angle without losing earlier versions. Also passes through the claim guardrail before being marked
ready to send.

### Admin
Password-protected. Six tabs:
- **Documents** -- upload new approved documents (auto-indexed and auto-classified into the Product
  Registry), assign/change a document's type, remove a document (runs in the background so the page
  isn't blocked), and manually trigger a full index rebuild.
- **Approved Claims** -- edit the small set of admin-typed facts (e.g. standard warranty terms) that
  supplement the approved documents in retrieval.
- **Restricted Claims** -- edit the policy the claim guardrail checks against: which categories
  (certifications, pricing, safety, delivery, etc.) require explicit source backing before being
  stated, and which are never allowed regardless of source.
- **Feedback** -- every Correct/Wrong/Unsafe click from the Assistant page, with accuracy trends over
  time and the ability to resolve or bulk-clear old entries.
- **Deals** -- every deal created across the app, with its linked requirements and Sales Aids visible
  inline.
- **Customer Requirements** -- customize the Customer Requirements form itself: edit a specific
  product's selectable fields, or add/edit/hide the generic questions asked of every deal.

### Claim guardrail (cross-cutting)
Any answer or generated document that touches a restricted category (certifications, pricing,
safety, delivery -- configured in Admin's Restricted Claims tab) is checked against the actual
approved source text before it's allowed to be shown as customer-ready. An unresolved risk or a
claim with no source backing marks the output "internal draft, needs review" rather than "ready to
send" -- this applies uniformly to the Assistant's customer-facing wording and to Sales Aids.

## How documents become answers

Uploading a document (via Admin, or by placing it in `data/approved_docs/` and reindexing) runs it
through:
1. **Extraction** -- format-specific parsing (DOCX, PPTX, PDF, XLSX, CSV, Markdown, plain text) into
   structured blocks (headings, paragraphs, tables), including OCR for embedded images and scanned
   PDF pages.
2. **Product Registry extraction** -- for catalogues specifically, the ordering-code nomenclature,
   selectable specs, and technical vocabulary are auto-extracted into a structured Product Index,
   which powers the deterministic catalog answers and the Customer Requirements product picker.
3. **Chunking + indexing** -- extracted text is chunked and embedded into a local vector index for
   retrieval.

Coverage and structure warnings (e.g. a table whose header may have been misdetected) surface in the
Admin Documents tab so a bad extraction doesn't silently ship.

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
4. Copy `.env.example` to `.env` and fill in the values below.

### Environment variables

| Variable | Required? | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes (if `LLM_PROVIDER=openai`, the default) | LLM calls for answer/discovery/sales-aid generation. |
| `ADMIN_PASSWORD` | Yes | Unlocks the Admin page. No default -- the app shows an explicit error if unset rather than accepting a guessable password. |
| `DATABASE_URL` | Only for cloud deployment | Postgres (Neon) connection string -- see [Persistence](#persistence). Use the *pooled* connection string. Without it, everything falls back to local files/SQLite. |
| `EMBEDDING_PROVIDER` | No (default `local`) | `openai` (uses `text-embedding-3-small` via the OpenAI API) or `local` (a local sentence-transformers model). Must match whatever the index was actually built with -- switching this after documents are indexed requires a full reindex. |
| `EMBEDDING_MODEL` | No | Overrides the default embedding model name. |
| `LLM_PROVIDER` | No (default `openai`) | `openai` or `azure_openai`. |
| `OPENAI_MODEL` | No (default `gpt-4.1-mini`) | Chat model name. |
| `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` / `AZURE_OPENAI_API_VERSION` | Only if `LLM_PROVIDER=azure_openai` | Azure OpenAI connection details. |
| `REQUIRE_SOURCES` | No (default `true`) | If true, skip the LLM call entirely (return the no-source message) when retrieval finds nothing relevant. |
| `ALLOW_CUSTOMER_FACING_OUTPUT` | No (default `false`) | Master switch for customer-facing wording/Sales Aids -- off by default regardless of the claim guardrail's own verdict. |
| `VECTOR_DB_PATH` | No (default `data/chroma_index`) | Where the local Chroma index lives. |
| `UI_THEME` | No (default `enterprise`) | `enterprise` (current dark theme + top-level Admin page) or `classic` (original layout). |

## Running it

```powershell
# Build the search index (run this once, and again whenever documents change outside the Admin UI)
python -m app.rag_pipeline --reindex

# Start the app
streamlit run app/streamlit_app.py
```

## Adding documents

Put approved files in `data/approved_docs/` (DOCX, PPTX, PDF, CSV, XLSX, Markdown, or text) and run
the reindex command above, or use the upload button on the Admin Documents tab, which extracts,
classifies, and indexes the file automatically and reports any extraction warnings.

## Persistence

Two independent stores, for two different reasons:

- **Postgres (Neon)** -- documents, Approved/Restricted Claims, deals, customer requirements,
  feedback, and generated Sales Aids, when `DATABASE_URL` is set. This is the durable source of
  truth for anything an admin or rep enters through the UI, since Streamlit Community Cloud's
  filesystem is wiped on every redeploy and on every cold start. Without `DATABASE_URL`, the same
  data lives in local files/SQLite under `data/` instead -- fully functional for local dev, just not
  durable across a cloud redeploy.
- **Local Chroma vector index** (`data/chroma_index/` by default) -- always local, never in
  Postgres. When `DATABASE_URL` is set, it's rebuilt automatically from Postgres the first time the
  app starts in a fresh process (cold start), so a redeploy always comes back with a correct index
  without a manual reindex step.

Don't run two Streamlit processes against the same local `data/chroma_index/` directory at the same
time -- concurrent access to the local Chroma index can corrupt it. If that happens, stop every
process touching it, delete `data/chroma_index/`, and restart one process; it rebuilds cleanly from
Postgres (or from `data/approved_docs/` via the reindex command, if not using Postgres).

## Deployment notes (Streamlit Community Cloud)

- The free tier sleeps the app after a period of inactivity, and the underlying Postgres (Neon free
  tier) also auto-suspends its compute after inactivity. The first visitor after either has slept
  triggers a full cold start -- pulling documents from Postgres, re-parsing and re-embedding all of
  them, and rebuilding the Product Registry -- which can take a couple of minutes on Community
  Cloud's shared hardware. Every subsequent visitor hits the already-warm process.
- `packages.txt` pins the apt-level system packages (`libgl1`, `libglib2.0-0t64`) required by the OCR
  dependency chain; without them, document extraction fails to import on a fresh container.
- Changing `requirements.txt` forces a full dependency reinstall on the next deploy, which takes
  noticeably longer than a code-only push.

## Known limitations

- No login/access control on the main pages (Assistant, Customer Requirements, Discovery Questions,
  Sales Aids) beyond whatever Streamlit Cloud's own app-visibility setting provides -- only the Admin
  page itself is password-protected. If the app's URL needs to stay restricted, that has to be
  configured at the Streamlit Cloud sharing-settings level, not in this codebase.
- Document extraction handles common table layouts well but can misread unusually complex tables
  (multiple header levels, or more than one logical section merged into a single physical table);
  the Admin Documents tab surfaces a warning when this looks like it happened, but review the source
  document if a table-heavy upload doesn't look right afterward.
- Background jobs (e.g. document removal) run one at a time within a single server process --
  there's no distributed job queue, so this assumes one Streamlit Cloud instance.

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
- `test_table_structure_check.py` -- `check_table_structure` (`app/document_loader.py`) catches a
  misdetected table header before it silently buries real spec data underneath it.
- `test_xlsx_extraction.py` -- XLSX title/table-region detection, including against the real approved
  Historical Sales workbook.

Most testing beyond this has been done by hand -- asking real questions in the app and checking the
answers, and (for admin features) exercising the actual UI end-to-end rather than only calling the
underlying functions directly. Broader automated coverage is still a work in progress.
