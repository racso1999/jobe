"""Classify and extract job-application details from an email using Claude.

A single ``messages.parse()`` call per email with a strict output schema, so the
model is constrained to valid JSON — no manual parsing/repair needed.
"""
import os
import json
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

import anthropic
from pydantic import BaseModel

# Haiku 4.5: fast and cheap, well-suited to this classification task.
# Note: Haiku 4.5 does not support the `effort` parameter, so none is set.
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")

_client: anthropic.Anthropic | None = None


class JobAnalysis(BaseModel):
    is_job_application: bool
    company: str
    job_title: str
    status: str
    confidence: float
    decision_reason: str
    paraphrase: str


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is required")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _email_context_to_text(email_context: dict[str, Any]) -> str:
    return json.dumps(email_context, ensure_ascii=True, indent=2)


def _parse_email_date(email_context: dict[str, Any]) -> datetime | None:
    date = str(email_context.get("date", ""))
    if not date:
        return None
    try:
        return parsedate_to_datetime(date)
    except (ValueError, TypeError):
        try:
            return datetime.fromisoformat(date.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None


SYSTEM_PROMPT = """You classify emails as job applications and extract details.

Rules:
- If the email is not related to a job application, set is_job_application to false.
- Do not invent details. If unknown, use "Unknown Company" or "Unknown Role".
- Keep status concise: one of Awaiting Response, Interview, Assessment, Offer,
  Rejected, or Other.
- confidence is your confidence from 0.0 to 1.0.
- decision_reason is a short justification.
- paraphrase is ONE short plain-language sentence summarising what this email
  says to the applicant (e.g. "They invited you to a first-round interview.").
  If not job-related, use an empty string."""


def analyze_job_application(email_context: dict[str, Any]) -> dict[str, Any]:
    """Single Claude call: classify an email and extract job application details.

    Returns a dict with the same shape the scanner expects (including a parsed
    ``applied_date`` derived from the email headers).
    """
    prompt = (
        "Determine whether the following email is related to a job application "
        "and extract the details.\n\nEmail Context (JSON):\n"
        f"{_email_context_to_text(email_context)}"
    )

    try:
        client = _get_client()
        response = client.messages.parse(
            model=MODEL_NAME,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_format=JobAnalysis,
        )
        parsed = response.parsed_output

        return {
            "is_job_application": bool(parsed.is_job_application),
            "company": parsed.company or "Unknown Company",
            "job_title": parsed.job_title or "Unknown Role",
            "status": parsed.status or "Other",
            "confidence": float(parsed.confidence or 0.0),
            "decision_reason": parsed.decision_reason,
            "paraphrase": parsed.paraphrase,
            "applied_date": _parse_email_date(email_context),
            "model": response.model,
        }
    except Exception as e:
        print(f"Anthropic analysis error: {e}")
        return {
            "is_job_application": False,
            "company": "Unknown Company",
            "job_title": "Unknown Role",
            "status": "Other",
            "confidence": 0.0,
            "decision_reason": f"Analyzer error: {e}",
            "paraphrase": "",
            "applied_date": _parse_email_date(email_context),
            "model": "",
        }
