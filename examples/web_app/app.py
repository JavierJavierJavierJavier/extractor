#!/usr/bin/env python3
# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streamlit app for file upload + prompt-based extraction with LangExtract."""

from __future__ import annotations

from collections import Counter
import io
import json
from typing import Any

import langextract as lx
from langextract import data_lib
import streamlit as st
import streamlit.components.v1 as components

try:
  from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
  PdfReader = None


DEFAULT_PROMPT = (
    "Extract entities in order of appearance. Use exact text spans from the "
    "source and add useful attributes when possible."
)

DEFAULT_EXAMPLES_JSON = json.dumps(
    [
        {
            "text": "Alice works at Acme Corp in Madrid.",
            "extractions": [
                {"class": "person", "text": "Alice"},
                {"class": "organization", "text": "Acme Corp"},
                {"class": "location", "text": "Madrid"},
            ],
        }
    ],
    indent=2,
)

DEFAULT_MODELS = {
    "Gemini": "gemini-2.5-flash",
    "OpenAI": "gpt-4o",
    "Ollama": "gemma2:2b",
    "Custom": "gemini-2.5-flash",
}


def _read_uploaded_file(uploaded_file: Any) -> str:
  """Read text content from txt/md/csv/json/pdf uploads."""
  file_name = uploaded_file.name.lower()
  raw_bytes = uploaded_file.getvalue()

  if file_name.endswith(".pdf"):
    if PdfReader is None:
      raise ValueError(
          "PDF support requires pypdf. Install dependencies from "
          "examples/web_app/requirements.txt."
      )
    reader = PdfReader(io.BytesIO(raw_bytes))
    page_texts = []
    for page in reader.pages:
      page_texts.append(page.extract_text() or "")
    text = "\n".join(page_texts).strip()
    if not text:
      raise ValueError(
          "The PDF was uploaded but no extractable text was found. "
          "Try an OCR-ready PDF or a text file."
      )
    return text

  for encoding in ("utf-8", "latin-1", "cp1252"):
    try:
      return raw_bytes.decode(encoding)
    except UnicodeDecodeError:
      continue
  raise ValueError(
      "Could not decode uploaded file as text. "
      "Try txt/md/csv/json/pdf formats."
  )


def _parse_examples_json(raw_json: str) -> list[lx.data.ExampleData]:
  """Parse user-provided few-shot examples from JSON."""
  try:
    payload = json.loads(raw_json)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Invalid JSON in examples field: {exc}") from exc

  if not isinstance(payload, list) or not payload:
    raise ValueError("Examples must be a non-empty JSON array.")

  examples: list[lx.data.ExampleData] = []
  for i, example_payload in enumerate(payload, start=1):
    if not isinstance(example_payload, dict):
      raise ValueError(f"Example #{i} must be an object.")

    example_text = example_payload.get("text")
    extractions_payload = example_payload.get("extractions")

    if not isinstance(example_text, str) or not example_text.strip():
      raise ValueError(f"Example #{i} needs a non-empty 'text' field.")
    if not isinstance(extractions_payload, list) or not extractions_payload:
      raise ValueError(
          f"Example #{i} needs a non-empty 'extractions' list."
      )

    extractions: list[lx.data.Extraction] = []
    for j, extraction_payload in enumerate(extractions_payload, start=1):
      if not isinstance(extraction_payload, dict):
        raise ValueError(
            f"Example #{i}, extraction #{j} must be an object."
        )

      extraction_class = extraction_payload.get(
          "class", extraction_payload.get("extraction_class")
      )
      extraction_text = extraction_payload.get(
          "text", extraction_payload.get("extraction_text")
      )
      attributes = extraction_payload.get("attributes")

      if not isinstance(extraction_class, str) or not extraction_class.strip():
        raise ValueError(
            f"Example #{i}, extraction #{j} needs 'class' "
            "or 'extraction_class'."
        )
      if not isinstance(extraction_text, str) or not extraction_text.strip():
        raise ValueError(
            f"Example #{i}, extraction #{j} needs 'text' "
            "or 'extraction_text'."
        )
      if attributes is not None and not isinstance(attributes, dict):
        raise ValueError(
            f"Example #{i}, extraction #{j} has invalid 'attributes'. "
            "Expected an object/dict."
        )

      extractions.append(
          lx.data.Extraction(
              extraction_class=extraction_class.strip(),
              extraction_text=extraction_text.strip(),
              attributes=attributes,
          )
      )

    examples.append(
        lx.data.ExampleData(
            text=example_text.strip(),
            extractions=extractions,
        )
    )

  return examples


def _resolve_input_text(
    uploaded_file: Any | None,
    manual_text: str,
    combine_sources: bool,
) -> tuple[str, str]:
  """Build the input text and explain where it came from."""
  manual_text = manual_text.strip()

  uploaded_text = ""
  if uploaded_file is not None:
    uploaded_text = _read_uploaded_file(uploaded_file).strip()

  if uploaded_text and manual_text and combine_sources:
    return f"{uploaded_text}\n\n{manual_text}", "uploaded file + manual text"
  if uploaded_text:
    return uploaded_text, "uploaded file"
  if manual_text:
    return manual_text, "manual text"

  raise ValueError(
      "Provide either an uploaded file or text in the manual text field."
  )


def _to_table_rows(document: lx.data.AnnotatedDocument) -> list[dict[str, Any]]:
  """Flatten extractions for a dataframe view."""
  rows = []
  for idx, extraction in enumerate(document.extractions or [], start=1):
    start_pos = None
    end_pos = None
    if extraction.char_interval:
      start_pos = extraction.char_interval.start_pos
      end_pos = extraction.char_interval.end_pos

    rows.append({
        "index": idx,
        "class": extraction.extraction_class,
        "text": extraction.extraction_text,
        "start_pos": start_pos,
        "end_pos": end_pos,
        "attributes": json.dumps(extraction.attributes or {}, ensure_ascii=False),
    })
  return rows


def _html_to_string(html_obj: Any) -> str:
  """Normalize lx.visualize return type to a plain HTML string."""
  if hasattr(html_obj, "data"):
    return html_obj.data
  return str(html_obj)


def main() -> None:
  st.set_page_config(
      page_title="LangExtract File Extractor",
      layout="wide",
  )

  st.title("LangExtract - file upload + prompt extraction")
  st.caption(
      "Upload a document, describe what to extract, provide few-shot examples, "
      "and run extraction in one place."
  )

  st.markdown(
      "LangExtract requires examples to guide extraction quality. "
      "Start with the default JSON template and adapt it to your task."
  )

  left_col, right_col = st.columns([1, 1], gap="large")

  with left_col:
    st.subheader("1) Input document")
    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt", "md", "csv", "json", "pdf"],
        help="Supported formats: txt, md, csv, json, pdf.",
    )

    manual_text = st.text_area(
        "Manual text (optional)",
        placeholder="Paste text here if you do not want to upload a file.",
        height=200,
    )
    combine_sources = st.checkbox(
        "Combine uploaded file and manual text",
        value=False,
    )

    st.subheader("2) Extraction instructions")
    prompt_description = st.text_area(
        "Prompt description",
        value=DEFAULT_PROMPT,
        height=120,
    )

    st.subheader("3) Few-shot examples (JSON)")
    examples_json = st.text_area(
        "Examples JSON",
        value=DEFAULT_EXAMPLES_JSON,
        height=260,
    )
    with st.expander("Expected examples JSON format"):
      st.code(
          """[
  {
    "text": "Alice works at Acme Corp in Madrid.",
    "extractions": [
      {"class": "person", "text": "Alice"},
      {"class": "organization", "text": "Acme Corp"},
      {"class": "location", "text": "Madrid"}
    ]
  }
]""",
          language="json",
      )

  with right_col:
    st.subheader("4) Model settings")
    provider = st.selectbox(
        "Provider preset",
        options=["Gemini", "OpenAI", "Ollama", "Custom"],
        help="Preset only affects defaults. You can still edit fields below.",
    )
    model_id = st.text_input(
        "model_id",
        value=DEFAULT_MODELS[provider],
        help="Examples: gemini-2.5-flash, gpt-4o, gemma2:2b.",
    )
    api_key = st.text_input(
        "API key (optional)",
        value="",
        type="password",
        help="If empty, LangExtract will try environment variables.",
    )
    model_url = st.text_input(
        "model_url (optional)",
        value="http://localhost:11434" if provider == "Ollama" else "",
        help="Use for local/self-hosted endpoints (for example Ollama).",
    )

    use_schema_default = provider not in ("OpenAI", "Ollama")
    use_schema_constraints = st.checkbox(
        "use_schema_constraints",
        value=use_schema_default,
    )
    fence_choice = st.selectbox(
        "fence_output",
        options=["auto", "true", "false"],
        index=0,
        help=(
            "Use 'auto' unless your provider requires a specific setting. "
            "OpenAI usually works with true + schema disabled."
        ),
    )

    temperature_enabled = st.checkbox("Set temperature manually", value=False)
    temperature = None
    if temperature_enabled:
      temperature = st.number_input(
          "temperature",
          min_value=0.0,
          max_value=2.0,
          value=0.0,
          step=0.1,
      )

    max_char_buffer = st.number_input(
        "max_char_buffer",
        min_value=200,
        max_value=5000,
        value=1000,
        step=100,
    )
    extraction_passes = st.number_input(
        "extraction_passes",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
    )
    max_workers = st.number_input(
        "max_workers",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    run = st.button("Run extraction", type="primary", use_container_width=True)

  if not run:
    return

  try:
    input_text, source_label = _resolve_input_text(
        uploaded_file=uploaded_file,
        manual_text=manual_text,
        combine_sources=combine_sources,
    )
    examples = _parse_examples_json(examples_json)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    st.error(str(exc))
    return

  extraction_kwargs: dict[str, Any] = {
      "text_or_documents": input_text,
      "prompt_description": prompt_description.strip(),
      "examples": examples,
      "model_id": model_id.strip(),
      "use_schema_constraints": use_schema_constraints,
      "max_char_buffer": int(max_char_buffer),
      "extraction_passes": int(extraction_passes),
      "max_workers": int(max_workers),
      "show_progress": False,
  }
  if api_key.strip():
    extraction_kwargs["api_key"] = api_key.strip()
  if model_url.strip():
    extraction_kwargs["model_url"] = model_url.strip()
  if temperature is not None:
    extraction_kwargs["temperature"] = float(temperature)
  if fence_choice != "auto":
    extraction_kwargs["fence_output"] = fence_choice == "true"

  with st.spinner("Running extraction..."):
    try:
      result = lx.extract(**extraction_kwargs)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      st.error(
          "Extraction failed. Check model settings, API key, and examples. "
          f"Details: {exc}"
      )
      return

  extraction_count = len(result.extractions or [])
  st.success(
      "Extraction completed from "
      f"{source_label}. Found {extraction_count} extraction(s)."
  )

  counts = Counter(ext.extraction_class for ext in (result.extractions or []))
  st.subheader("Summary")
  st.json(dict(counts))

  st.subheader("Extractions table")
  rows = _to_table_rows(result)
  if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
  else:
    st.info("No extractions were returned.")

  st.subheader("Interactive visualization")
  html_content = _html_to_string(lx.visualize(result))
  components.html(html_content, height=700, scrolling=True)

  doc_dict = data_lib.annotated_document_to_dict(result)
  pretty_json = json.dumps(doc_dict, ensure_ascii=False, indent=2)
  jsonl_line = json.dumps(doc_dict, ensure_ascii=False) + "\n"

  download_col_1, download_col_2, download_col_3 = st.columns(3)
  with download_col_1:
    st.download_button(
        "Download JSON",
        data=pretty_json.encode("utf-8"),
        file_name="langextract_result.json",
        mime="application/json",
        use_container_width=True,
    )
  with download_col_2:
    st.download_button(
        "Download JSONL",
        data=jsonl_line.encode("utf-8"),
        file_name="langextract_result.jsonl",
        mime="application/x-ndjson",
        use_container_width=True,
    )
  with download_col_3:
    st.download_button(
        "Download HTML visualization",
        data=html_content.encode("utf-8"),
        file_name="langextract_visualization.html",
        mime="text/html",
        use_container_width=True,
    )

  with st.expander("Raw result JSON"):
    st.code(pretty_json, language="json")


if __name__ == "__main__":
  main()
