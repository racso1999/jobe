import os
import json
import google.generativeai as genai
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any


MODEL_NAME = "gemini-1.5-flash-latest"
MODEL_CANDIDATES = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]
_WORKING_MODEL_NAME: str | None = None


def _email_context_to_text(email_context: dict[str, Any]) -> str:
    """Convert full email context to a JSON string for Gemini prompts."""
    return json.dumps(email_context, ensure_ascii=True, indent=2)


def _extract_json_object(text: str) -> dict | None:
    """Extract a JSON object from raw model output."""
    cleaned = text.strip()

    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON object in mixed text output.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def configure_gemini():
    """Configure Gemini API with the API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required for Gemini parser")
    genai.configure(api_key=api_key)


def _extract_model_basename(model_name: str) -> str:
    return model_name.split("/", 1)[1] if model_name.startswith("models/") else model_name


def _build_model_candidates() -> list[str]:
    configured_model = os.environ.get("GEMINI_MODEL", "").strip()
    names: list[str] = []

    if _WORKING_MODEL_NAME:
        names.append(_WORKING_MODEL_NAME)
    if configured_model:
        names.append(configured_model)
    names.extend(MODEL_CANDIDATES)

    # Deduplicate while preserving order.
    deduped: list[str] = []
    for name in names:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def _generate_content_with_fallback(prompt: str):
    """Generate content by trying available model names until one succeeds."""
    global _WORKING_MODEL_NAME

    last_error: Exception | None = None
    for model_name in _build_model_candidates():
        for candidate in (model_name, f"models/{_extract_model_basename(model_name)}"):
            try:
                model = genai.GenerativeModel(candidate)
                response = model.generate_content(prompt)
                _WORKING_MODEL_NAME = _extract_model_basename(candidate)
                return response, _WORKING_MODEL_NAME
            except Exception as exc:
                last_error = exc

    raise RuntimeError(f"No working Gemini model found from candidates: {_build_model_candidates()}. Last error: {last_error}")


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


def analyze_job_application_with_gemini(email_context: dict[str, Any]) -> dict[str, Any]:
    """Single Gemini call: classify email and extract job application details."""
    prompt = f"""
Determine whether the email is related to a job application and fetch the details.

Email Context (JSON):
{_email_context_to_text(email_context)}

Return ONLY valid JSON with this exact schema:
{{
  "is_job_application": true,
  "company": "Company Name",
  "job_title": "Job Title",
  "status": "Awaiting Response",
  "confidence": 0.0,
  "decision_reason": "Short reason"
}}

Rules:
- If the email is not job-related, set is_job_application to false.
- Do not invent details. If unknown, use "Unknown Company" or "Unknown Role".
- Keep status concise: Awaiting Response, Interview, Assessment, Offer, Rejected, or Other.
"""

    try:
        configure_gemini()
        response, used_model = _generate_content_with_fallback(prompt)
        response_text = (response.text or "").strip()
        parsed = _extract_json_object(response_text)

        if parsed is None:
            return {
                "is_job_application": False,
                "company": "Unknown Company",
                "job_title": "Unknown Role",
                "status": "Other",
                "confidence": 0.0,
                "decision_reason": "Gemini returned non-JSON output.",
                "applied_date": _parse_email_date(email_context),
                "raw_response": response_text,
                "model": used_model,
            }

        return {
            "is_job_application": bool(parsed.get("is_job_application", False)),
            "company": str(parsed.get("company") or "Unknown Company"),
            "job_title": str(parsed.get("job_title") or "Unknown Role"),
            "status": str(parsed.get("status") or "Other"),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "decision_reason": str(parsed.get("decision_reason", "")),
            "applied_date": _parse_email_date(email_context),
            "raw_response": response_text,
            "model": used_model,
        }
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return {
            "is_job_application": False,
            "company": "Unknown Company",
            "job_title": "Unknown Role",
            "status": "Other",
            "confidence": 0.0,
            "decision_reason": f"Analyzer error: {e}",
            "applied_date": _parse_email_date(email_context),
            "raw_response": "",
            "model": "",
        }


def parse_job_application_with_gemini(
    email_context: dict[str, Any],
) -> dict | None:
    """
    Use Gemini API to parse a job application email.
    
    Returns:
        dict with keys: company, job_title, status, applied_date
        or None if parsing fails or email is not job-related
    """
    analysis = analyze_job_application_with_gemini(email_context)
    if not analysis.get("is_job_application"):
        return None
    return {
        "company": analysis.get("company", "Unknown Company"),
        "job_title": analysis.get("job_title", "Unknown Role"),
        "status": analysis.get("status", "Other"),
        "applied_date": analysis.get("applied_date"),
        "confidence": analysis.get("confidence", 0.0),
    }


def classify_job_application_with_gemini(email_context: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible wrapper using the single Gemini analysis call."""
    analysis = analyze_job_application_with_gemini(email_context)
    return {
        "is_job_application": bool(analysis.get("is_job_application", False)),
        "confidence": float(analysis.get("confidence", 0.0) or 0.0),
        "decision_reason": str(analysis.get("decision_reason", "")),
        "evidence": [],
        "raw_response": str(analysis.get("raw_response", "")),
        "prompt": "",
        "model": str(analysis.get("model", "")),
    }


def is_job_application_email_with_gemini(email_context: dict[str, Any]) -> bool:
    """
    Use Gemini to determine if an email is job-related.
    
    Returns True if it's job-related, False otherwise.
    """
    result = classify_job_application_with_gemini(email_context)
    return bool(result.get("is_job_application", False))
