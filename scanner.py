"""Reusable inbox scan engine shared by the web route and the background job.

The scanner is deliberately session-free: it takes a persisted ``UserAccount``
and a DB session, so the hourly ``BackgroundScheduler`` job can run it without
any HTTP request in scope.
"""
import os
import re
import json
import base64
from datetime import datetime, timedelta
from email.utils import parseaddr

from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sqlalchemy import func
from sqlalchemy.orm import Session

import scan_status
from database import UserAccount, JobApplication, ProcessedEmail, JobEmail
from anthropic_parser import analyze_job_application

ONBOARD_DAYS = int(os.environ.get("ONBOARD_DAYS", "30"))
# Small look-back so an email arriving mid-scan isn't missed between ticks.
OVERLAP_SECONDS = int(os.environ.get("SCAN_OVERLAP_SECONDS", "300"))
# Bound cost of a single scan (esp. the 30-day onboarding sweep).
MAX_MESSAGES_PER_SCAN = int(os.environ.get("MAX_MESSAGES_PER_SCAN", "500"))
# Same company + this much title-word overlap => same application (else new row).
TITLE_MATCH_THRESHOLD = float(os.environ.get("TITLE_MATCH_THRESHOLD", "0.75"))


# ---------------------------------------------------------------------------
# Email parsing helpers (moved from main.py so both callers can share them)
# ---------------------------------------------------------------------------


def _decode_base64url(data: str) -> str:
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_email_body(payload: dict) -> str:
    """Extract combined text body content from all payload parts."""
    if not payload:
        return ""

    fragments: list[str] = []
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")
    if body_data and mime_type in {"text/plain", "text/html"}:
        text = _decode_base64url(body_data)
        if mime_type == "text/html":
            text = re.sub(r"<[^>]+>", " ", text)
        normalized = " ".join(text.split())
        if normalized:
            fragments.append(normalized)

    for part in payload.get("parts", []):
        part_text = _extract_email_body(part)
        if part_text:
            fragments.append(part_text)

    return "\n\n".join(fragments)


def _sender_name(from_email: str) -> str:
    """Friendly sender label for the status line: display name, else company."""
    name, _addr = parseaddr(from_email or "")
    name = name.strip().strip('"')
    return name or _fallback_company(from_email)


def _fallback_company(from_email: str) -> str:
    match = re.search(r"@([A-Za-z0-9.-]+)", from_email or "")
    if not match:
        return "Unknown Company"
    domain = match.group(1).lower()
    parts = [p for p in domain.split(".") if p and p not in {"com", "co", "org", "net", "io", "ai"}]
    return parts[-1].capitalize() if parts else "Unknown Company"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


# Interchangeable role words collapse to one token so e.g. "Software Developer"
# and "Software Engineer" are recognised as the same role.
_ROLE_SYNONYMS = {
    "developer": "engineer",
    "dev": "engineer",
    "programmer": "engineer",
    "coder": "engineer",
    "swe": "engineer",
    "sde": "engineer",
}


def _title_tokens(title: str) -> set:
    """Lowercased word tokens of a job title, with role-word synonyms collapsed."""
    tokens = re.findall(r"[a-z0-9]+", (title or "").lower())
    return {_ROLE_SYNONYMS.get(t, t) for t in tokens}


def _title_coverage(a: str, b: str) -> float:
    """Token coverage: shared words / the shorter title's word count (0..1)."""
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _find_matching_job(db: Session, user_email: str, company: str, job_title: str,
                       exclude_id: int | None = None):
    """Existing job for the same company whose title best covers `job_title`
    at >= TITLE_MATCH_THRESHOLD, else None (=> new row). `exclude_id` skips a
    row (used when a manual edit shouldn't match itself)."""
    candidates = (
        db.query(JobApplication)
        .filter(
            JobApplication.user_email == user_email,
            func.lower(JobApplication.company) == company.lower(),
        )
        .all()
    )
    best, best_score = None, 0.0
    for job in candidates:
        if exclude_id is not None and job.id == exclude_id:
            continue
        score = _title_coverage(job.job_title, job_title)
        if score > best_score:
            best, best_score = job, score
    return best if best_score >= TITLE_MATCH_THRESHOLD else None


def credentials_from_account(account: UserAccount) -> Credentials:
    return Credentials(
        token=account.token,
        refresh_token=account.refresh_token,
        token_uri=account.token_uri,
        client_id=account.client_id,
        client_secret=account.client_secret,
        scopes=json.loads(account.scopes) if account.scopes else None,
    )


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------


def _build_query(account: UserAccount) -> str:
    """Gmail search query limiting the scan to the relevant time window."""
    if account.last_scanned_at is None:
        after = datetime.utcnow() - timedelta(days=ONBOARD_DAYS)
    else:
        after = account.last_scanned_at - timedelta(seconds=OVERLAP_SECONDS)
    return f"after:{int(after.timestamp())}"


def _list_message_ids(service, query: str) -> list[str]:
    """List inbox message ids matching the query, following pagination."""
    ids: list[str] = []
    page_token: str | None = None
    while True:
        results = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=["INBOX"],
                q=query,
                maxResults=100,
                pageToken=page_token,
            )
            .execute()
        )
        ids.extend(m["id"] for m in results.get("messages", []))
        page_token = results.get("nextPageToken")
        if not page_token or len(ids) >= MAX_MESSAGES_PER_SCAN:
            break
    return ids[:MAX_MESSAGES_PER_SCAN]


def scan_account(db: Session, account: UserAccount) -> dict:
    """Scan one account's inbox, grading and storing only unseen emails.

    Returns a small stats dict for logging: how many messages the window
    contained, how many were skipped as already-processed, and how many new
    job applications were stored.
    """
    credentials = credentials_from_account(account)

    # Credentials rebuilt from the DB carry no expiry, so `expired` is
    # unreliable and the stored access token is usually stale by the next
    # hourly tick. Refresh unconditionally when we hold a refresh token, and
    # persist the fresh access token for future runs.
    if credentials.refresh_token:
        credentials.refresh(GoogleRequest())
        account.token = credentials.token
        db.commit()

    service = build("gmail", "v1", credentials=credentials)

    # Stamp the scan start time up front so emails arriving during the scan are
    # still caught next tick (thanks to OVERLAP_SECONDS).
    scan_started = datetime.utcnow()

    scan_status.set_status("Checking your inbox…")
    query = _build_query(account)
    message_ids = _list_message_ids(service, query)

    # Skip anything already processed without fetching or grading it.
    already = {
        row.email_id
        for row in db.query(ProcessedEmail.email_id)
        .filter(ProcessedEmail.email_id.in_(message_ids))
        .all()
    } if message_ids else set()

    stats = {"found": len(message_ids), "skipped": 0, "new_jobs": 0, "parsed": 0}

    for msg_id in message_ids:
        if msg_id in already:
            stats["skipped"] += 1
            continue

        detail = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )
        raw_headers = detail.get("payload", {}).get("headers", [])
        headers = {h.get("name", ""): h.get("value", "") for h in raw_headers}
        subject = headers.get("Subject", "(no subject)")
        from_email = headers.get("From", "")
        date = headers.get("Date", "")
        snippet = detail.get("snippet", "")
        body = _extract_email_body(detail.get("payload", {})) or snippet
        internal_date = detail.get("internalDate", "")

        email_context = {
            "message_id": detail.get("id", ""),
            "thread_id": detail.get("threadId", ""),
            "internal_date": internal_date,
            "label_ids": detail.get("labelIds", []),
            "snippet": snippet,
            "headers": headers,
            "from": from_email,
            "to": headers.get("To", ""),
            "cc": headers.get("Cc", ""),
            "reply_to": headers.get("Reply-To", ""),
            "subject": subject,
            "date": date,
            "body": body,
            "mime_type": detail.get("payload", {}).get("mimeType", ""),
        }

        scan_status.set_status(f"Parsing email from {_sender_name(from_email)}…")
        analysis = analyze_job_application(email_context)
        stats["parsed"] += 1
        is_job = bool(analysis.get("is_job_application", False))

        # Record that we've handled this email so it's never parsed again.
        db.add(
            ProcessedEmail(
                email_id=msg_id,
                user_email=account.email,
                internal_date=internal_date,
                is_job_application=is_job,
            )
        )

        if is_job:
            company = analysis.get("company") or _fallback_company(from_email)
            job_title = analysis.get("job_title") or "Unknown Role"
            status = analysis.get("status") or "Other"
            applied_date = analysis.get("applied_date")

            # Same company + >=75% title-word overlap => same application, so a
            # follow-up email updates that entry (and moves it to the top);
            # otherwise it's a genuinely different role and gets its own row.
            existing = _find_matching_job(db, account.email, company, job_title)

            if existing:
                existing.status = status or existing.status
                existing.email_id = msg_id  # remember the most recent email
                existing.email_subject = subject
                existing.email_body = body
                if applied_date:
                    existing.applied_date = applied_date
                # Bump so the freshly-updated job sorts to the top.
                existing.updated_at = datetime.utcnow()
                job = existing
            else:
                job = JobApplication(
                    user_email=account.email,
                    company=company,
                    job_title=job_title,
                    status=status,
                    email_id=msg_id,
                    email_subject=subject,
                    email_body=body,
                    applied_date=applied_date,
                )
                db.add(job)
                stats["new_jobs"] += 1

            job.unread = True  # flag for the dashboard bell until the user views it
            db.flush()  # ensure job.id is available for the JobEmail link

            # When the email was received (Gmail internalDate is ms since epoch).
            received_at = applied_date
            if internal_date:
                try:
                    received_at = datetime.utcfromtimestamp(int(internal_date) / 1000)
                except (ValueError, TypeError):
                    received_at = applied_date

            db.add(
                JobEmail(
                    job_id=job.id,
                    user_email=account.email,
                    email_id=msg_id,
                    message_ref=headers.get("Message-ID") or headers.get("Message-Id") or "",
                    subject=subject,
                    paraphrase=analysis.get("paraphrase") or "",
                    received_at=received_at,
                )
            )
            scan_status.bump_revision()  # signal the dashboard there's new data

        db.commit()

    account.last_scanned_at = scan_started
    db.commit()

    return stats
