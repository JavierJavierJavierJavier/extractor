# LangExtract Streamlit Web App

This example app lets you:

1. Upload a document (`txt`, `md`, `csv`, `json`, `pdf`)
2. Or load a document from a local file path (useful for very large files)
3. Write a prompt describing what to extract
4. Provide few-shot examples in JSON format
5. Run `langextract` and review/download the results

## Quick start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install LangExtract from source
pip install -e .

# If you want to use OpenAI models:
# pip install -e ".[openai]"

# Install app dependencies
pip install -r examples/web_app/requirements.txt
```

Set your key for cloud models:

```bash
export LANGEXTRACT_API_KEY="your-api-key"
```

Run the app:

```bash
python3 -m streamlit run examples/web_app/app.py
```

## Large files

If Streamlit upload limits block large files, use the **Local file path** field
inside the app. This bypasses browser upload limits because the app reads the
file directly from disk.

## Provider notes

- **Gemini**: usually works with `use_schema_constraints=True`
- **OpenAI**: usually requires `use_schema_constraints=False` and `fence_output=true`
- **Ollama**: typically local endpoint `http://localhost:11434`, with
  `use_schema_constraints=False`

## Few-shot example format

```json
[
  {
    "text": "Alice works at Acme Corp in Madrid.",
    "extractions": [
      {"class": "person", "text": "Alice"},
      {"class": "organization", "text": "Acme Corp"},
      {"class": "location", "text": "Madrid"}
    ]
  }
]
```

You can also use keys `extraction_class` and `extraction_text` instead of
`class` and `text`.
