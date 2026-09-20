"""Tests for PIIRedactor."""

import pytest
from scholar_rag.redactor import PIIRedactor


@pytest.fixture
def redactor():
    return PIIRedactor()


def test_redact_email(redactor):
    text, found = redactor.redact("Contact john@example.com for info")
    assert "[EMAIL REDACTED]" in text
    assert found is True


def test_redact_orcid_uri(redactor):
    text, found = redactor.redact("Author: https://orcid.org/0000-0002-1825-0097")
    assert "[ORCID REDACTED]" in text
    assert found is True


def test_redact_orcid_numeric(redactor):
    text, found = redactor.redact("ORCID: 0000-0002-1825-0097")
    assert "[ORCID REDACTED]" in text
    assert found is True


def test_redact_phone(redactor):
    text, found = redactor.redact("Call (555) 123-4567")
    assert "[PHONE REDACTED]" in text
    assert found is True


def test_redact_grant(redactor):
    text, found = redactor.redact("NIH Grant R01-12345")
    assert "[GRANT REDACTED]" in text
    assert found is True


def test_no_pii_found(redactor):
    text, found = redactor.redact("No PII here")
    assert text == "No PII here"
    assert found is False


def test_redact_dict_recursive(redactor):
    data = {"author": "Email me at test@uni.edu", "title": "Clean"}
    result, found = redactor.redact_dict(data)
    assert "[EMAIL REDACTED]" in result["author"]
    assert result["title"] == "Clean"
    assert found is True


def test_redact_list_recursive(redactor):
    items = ["Contact a@b.com", "Normal text"]
    result, found = redactor.redact_list(items)
    assert "[EMAIL REDACTED]" in result[0]
    assert result[1] == "Normal text"
    assert found is True


def test_multiple_pii_types(redactor):
    text, found = redactor.redact("Email a@b.com or call 555-1234")
    assert found is True
    assert "[EMAIL REDACTED]" in text
