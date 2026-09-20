"""LLM-based structured extraction for academic papers."""

from __future__ import annotations

import json
import re
from typing import Any

from .redactor import PIIRedactor
from .schemas import ExtractionResult, PaperExtraction, AuthorExtraction


class LLMExtractor:
    """Extract structured metadata from academic text using LLM."""

    EXTRACTION_PROMPT = """Extract structured metadata from the following academic text.
Return a JSON object with these fields:
- title: Paper title
- authors: List of author objects with family_name, given_name, orcid, affiliation
- abstract: Paper abstract
- year: Publication year (integer)
- doi: DOI identifier
- venue: Publication venue
- keywords: List of keywords
- contributions: Key contributions
- limitations: Stated limitations
- future_work: Future work directions

Text:
{text}

Return ONLY valid JSON, no markdown fences."""

    _FIELD_WEIGHTS = {
        "title": 0.15,
        "authors": 0.10,
        "abstract": 0.15,
        "year": 0.10,
        "doi": 0.10,
        "venue": 0.10,
        "keywords": 0.10,
        "contributions": 0.10,
        "limitations": 0.05,
        "future_work": 0.05,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-pro",
        temperature: float = 0.1,
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._redactor = PIIRedactor()

    async def extract_from_text(
        self,
        text: str,
        schema: type | None = None,
        chunk_index: int | None = None,
        source_file: str | None = None,
        section_name: str | None = None,
    ) -> ExtractionResult:
        """Extract structured metadata from text."""
        try:
            response = await self._call_llm(text)
            extracted_data = self._parse_llm_json(response)
        except Exception:
            # Fallback to heuristic extraction
            extracted_data = self._heuristic_extract(text)

        # Apply PII redaction
        extracted_data, pii_found = self._redactor.redact_dict(extracted_data)

        # Build extraction model
        extraction = PaperExtraction(**extracted_data)

        # Calculate confidence
        confidence = self._calculate_confidence(extracted_data)

        return ExtractionResult(
            extraction=extraction,
            confidence=confidence,
            source_file=source_file,
            chunk_index=chunk_index,
            pii_redacted=pii_found,
        )

    async def _call_llm(self, text: str) -> str:
        """Call Gemini REST API."""
        import httpx

        prompt = self.EXTRACTION_PROMPT.format(text=text[:8000])  # Limit text length

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self._temperature},
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(url, headers=headers, params={"key": self._api_key}, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def _parse_llm_json(self, response: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown fences."""
        # Remove markdown code fences
        response = re.sub(r"```json?\s*", "", response)
        response = re.sub(r"```\s*$", "", response)
        response = response.strip()

        return json.loads(response)

    def _heuristic_extract(self, text: str) -> dict[str, Any]:
        """Rule-based fallback extraction."""
        result: dict[str, Any] = {
            "title": None,
            "authors": [],
            "abstract": None,
            "year": None,
            "doi": None,
            "venue": None,
            "keywords": [],
            "contributions": [],
            "limitations": [],
            "future_work": [],
        }

        # Extract title (first heading)
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if title_match:
            result["title"] = title_match.group(1).strip()

        # Extract DOI
        doi_match = re.search(r"10\.\d{4,}/[^\s]+", text)
        if doi_match:
            result["doi"] = doi_match.group(0).rstrip(".,;")

        # Extract year
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        if year_match:
            result["year"] = int(year_match.group(0))

        # Extract abstract (between "abstract" and next heading)
        abstract_match = re.search(r"(?:abstract|Abstract)[:\s]*\n(.+?)(?:\n#|\n\*\*|\Z)", text, re.DOTALL)
        if abstract_match:
            result["abstract"] = abstract_match.group(1).strip()[:2000]

        return result

    def _calculate_confidence(self, extracted: dict[str, Any]) -> float:
        """Calculate weighted confidence score."""
        score = 0.0
        for field, weight in self._FIELD_WEIGHTS.items():
            value = extracted.get(field)
            if value is not None and value != [] and value != "":
                score += weight
        return round(score, 3)
