import os
import secrets
import base64
import re
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from starlette.middleware.sessions import SessionMiddleware
from database import init_db, SessionLocal, JobApplication
from gemini_parser import analyze_job_application_with_gemini

load_dotenv()

# Allow HTTP for local development only — remove in production
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

SESSION_SECRET = os.environ.get("SESSION_SECRET")
if not SESSION_SECRET:
    raise RuntimeError("SESSION_SECRET environment variable is required")

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    raise RuntimeError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are required")

REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://localhost:8000/auth/callback")

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=86400)
templates = Jinja2Templates(directory="templates")

# Initialize database on startup
init_db()


def _build_flow() -> Flow:
    return Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [REDIRECT_URI],
            }
        },
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )


def _credentials_from_session(data: dict) -> Credentials:
    return Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data["scopes"],
    )


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


def _fallback_company(from_email: str) -> str:
    match = re.search(r"@([A-Za-z0-9.-]+)", from_email or "")
    if not match:
        return "Unknown Company"
    domain = match.group(1).lower()
    parts = [p for p in domain.split(".") if p and p not in {"com", "co", "org", "net", "io", "ai"}]
    return parts[-1].capitalize() if parts else "Unknown Company"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = request.session.get("user")
    if not user:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request, "user": user},
        )
    # If logged in, redirect to jobs page
    return RedirectResponse("/jobs")


@app.get("/auth/login")
async def login(request: Request):
    flow = _build_flow()
    state = secrets.token_urlsafe(32)
    request.session["oauth_state"] = state
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        state=state,
        prompt="consent",
    )
    request.session["oauth_code_verifier"] = flow.code_verifier
    return RedirectResponse(auth_url)


@app.get("/auth/callback")
async def callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
):
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorisation code")

    stored_state = request.session.get("oauth_state")
    code_verifier = request.session.get("oauth_code_verifier")
    if not state or not stored_state or state != stored_state:
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    if not code_verifier:
        raise HTTPException(status_code=400, detail="Missing OAuth code verifier")

    request.session.pop("oauth_state", None)
    request.session.pop("oauth_code_verifier", None)

    flow = _build_flow()
    flow.code_verifier = code_verifier
    flow.fetch_token(code=code)
    credentials = flow.credentials

    user_info_svc = build("oauth2", "v2", credentials=credentials)
    user_info = user_info_svc.userinfo().get().execute()

    request.session["credentials"] = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": list(credentials.scopes or SCOPES),
    }
    request.session["user"] = {
        "email": user_info.get("email"),
        "name": user_info.get("name"),
        "picture": user_info.get("picture"),
    }

    return RedirectResponse("/")


@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


@app.get("/jobs", response_class=HTMLResponse)
async def jobs(request: Request):
    user = request.session.get("user")
    creds_data = request.session.get("credentials")

    if not user:
        return RedirectResponse("/auth/login")

    jobs_list = []

    # Scan inbox and classify each email.
    if creds_data:
        credentials = _credentials_from_session(creds_data)

        # Refresh if expired
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(GoogleRequest())
                request.session["credentials"]["token"] = credentials.token
            except Exception:
                request.session.clear()
                return RedirectResponse("/auth/login")

        service = build("gmail", "v1", credentials=credentials)

        try:
            results = (
                service.users()
                .messages()
                .list(userId="me", maxResults=50, labelIds=["INBOX"])
                .execute()
            )

            messages = results.get("messages", [])
            db = SessionLocal()
            try:
                for msg in messages:
                    detail = (
                        service.users()
                        .messages()
                        .get(
                            userId="me",
                            id=msg["id"],
                            format="full",
                        )
                        .execute()
                    )
                    raw_headers = detail.get("payload", {}).get("headers", [])
                    headers = {h.get("name", ""): h.get("value", "") for h in raw_headers}
                    subject = headers.get("Subject", "(no subject)")
                    from_email = headers.get("From", "")
                    date = headers.get("Date", "")
                    snippet = detail.get("snippet", "")
                    body = _extract_email_body(detail.get("payload", {})) or snippet

                    email_context = {
                        "message_id": detail.get("id", ""),
                        "thread_id": detail.get("threadId", ""),
                        "internal_date": detail.get("internalDate", ""),
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

                    analysis = analyze_job_application_with_gemini(email_context)
                    if not bool(analysis.get("is_job_application", False)):
                        continue

                    parsed = {
                        "company": analysis.get("company") or _fallback_company(from_email),
                        "job_title": analysis.get("job_title") or "Unknown Role",
                        "status": analysis.get("status") or "Other",
                        "applied_date": analysis.get("applied_date"),
                    }

                    existing = db.query(JobApplication).filter(
                        JobApplication.email_id == msg["id"],
                        JobApplication.user_email == user["email"],
                    ).first()

                    if existing:
                        existing.company = parsed.get("company") or existing.company
                        existing.job_title = parsed.get("job_title") or existing.job_title
                        existing.status = parsed.get("status") or existing.status
                        existing.email_subject = subject
                        existing.email_body = body
                        if parsed.get("applied_date"):
                            existing.applied_date = parsed.get("applied_date")
                    else:
                        db.add(
                            JobApplication(
                                user_email=user["email"],
                                company=parsed.get("company", "Unknown Company"),
                                job_title=parsed.get("job_title", "Job Application"),
                                status=parsed.get("status", "Awaiting Response"),
                                email_id=msg["id"],
                                email_subject=subject,
                                email_body=body,
                                applied_date=parsed.get("applied_date"),
                            )
                        )

                db.commit()

                saved_jobs = db.query(JobApplication).filter(
                    JobApplication.user_email == user["email"]
                ).order_by(JobApplication.created_at.desc()).all()

                jobs_list = [
                    {
                        "company": job.company,
                        "job_title": job.job_title,
                        "status": job.status,
                        "date": (job.applied_date or job.created_at).strftime("%Y-%m-%d")
                        if (job.applied_date or job.created_at)
                        else "",
                    }
                    for job in saved_jobs
                ]
            finally:
                db.close()
        except Exception as e:
            print(f"Error scanning emails: {e}")

    return templates.TemplateResponse(
        request=request,
        name="jobs.html",
        context={"request": request, "user": user, "jobs": jobs_list},
    )
