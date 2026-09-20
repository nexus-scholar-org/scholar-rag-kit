"""PII detection and redaction for extracted content."""

from __future__ import annotations

import re
from typing import Any


class PIIRedactor:
    """Regex-based PII detection and redaction."""

    # Compiled regex patterns
    EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    ORCID_URI_PATTERN = re.compile(r"https?://orcid\.org/\d{4}-\d{4}-\d{4}-\d{3}[\dX]")
    ORCID_NUMERIC_PATTERN = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]")
    PHONE_PATTERN = re.compile(r"[\+]?[(]?\d{1,4}[)]?[-\s.]?\d{1,4}[-\s.]?\d{1,9}")
    GRANT_PATTERN = re.compile(
        r"(?:NIH|NSF|ERC|DFG|grant\s+(?:number\s+)?)[\s:]?[A-Z0-9\-]+",
        re.IGNORECASE,
    )

    def redact(self, text: str) -> tuple[str, bool]:
        """Redact PII from text. Returns (redacted_text, pii_found)."""
        pii_found = False

        # Apply redactions (order matters: URI before numeric ORCID)
        text, count = self.ORCID_URI_PATTERN.subn("[ORCID REDACTED]", text)
        pii_found = pii_found or count > 0

        text, count = self.EMAIL_PATTERN.subn("[EMAIL REDACTED]", text)
        pii_found = pii_found or count > 0

        text, count = self.ORCID_NUMERIC_PATTERN.subn("[ORCID REDACTED]", text)
        pii_found = pii_found or count > 0

        text, count = self.PHONE_PATTERN.subn("[PHONE REDACTED]", text)
        pii_found = pii_found or count > 0

        text, count = self.GRANT_PATTERN.subn("[GRANT REDACTED]", text)
        pii_found = pii_found or count > 0

        return text, pii_found

    def redact_dict(self, data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Recursively redact PII from a dictionary."""
        pii_found = False
        result = {}

        for key, value in data.items():
            if isinstance(value, str):
                value, found = self.redact(value)
                pii_found = pii_found or found
            elif isinstance(value, dict):
                value, found = self.redact_dict(value)
                pii_found = pii_found or found
            elif isinstance(value, list):
                value, found = self.redact_list(value)
                pii_found = pii_found or found

            result[key] = value

        return result, pii_found

    def redact_list(self, items: list[Any]) -> tuple[list[Any], bool]:
        """Recursively redact PII from a list."""
        pii_found = False
        result = []

        for item in items:
            if isinstance(item, str):
                item, found = self.redact(item)
                pii_found = pii_found or found
            elif isinstance(item, dict):
                item, found = self.redact_dict(item)
                pii_found = pii_found or found
            elif isinstance(item, list):
                item, found = self.redact_list(item)
                pii_found = pii_found or found

            result.append(item)

        return result, pii_found
