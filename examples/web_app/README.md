# LangExtract Streamlit Web App

This example app lets you:

1. Upload a document (`txt`, `md`, `csv`, `json`, `pdf`)
2. Write a prompt describing what to extract
3. Provide few-shot examples in JSON format
4. Run `langextract` and review/download the results

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
