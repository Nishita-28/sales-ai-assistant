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

## Rules

- Only put approved, non-confidential documents in `data/approved_docs/`. No chip design, customer,
  legal, financial, or vendor documents.
- Never commit the real `.env` file or an API key.
- This is internal-only. It is not meant to be exposed publicly or connected to customers directly.

## Tests

There are no automated tests yet. Testing so far has been done by hand — asking real questions in the
app and checking the answers. Adding real test files under `tests/` is still to do.
