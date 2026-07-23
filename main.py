import os
import secrets
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from starlette.middleware.sessions import SessionMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from database import (
    init_db, SessionLocal, JobApplication, JobEmail, ProcessedEmail,
    UserAccount, upsert_account,
)
from scanner import scan_account
import scan_status

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

SCAN_INTERVAL_MINUTES = float(os.environ.get("SCAN_INTERVAL_MINUTES", "1"))

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]

scheduler = BackgroundScheduler()


def scan_all_accounts():
    """Background job: incrementally scan every persisted account's inbox."""
    db = SessionLocal()
    try:
        accounts = db.query(UserAccount).filter(UserAccount.refresh_token.isnot(None)).all()
        for account in accounts:
            try:
                stats = scan_account(db, account)
                print(f"[scan] {account.email}: {stats}")
            except Exception as e:  # keep scanning other accounts on failure
                db.rollback()
                print(f"[scan] error for {account.email}: {e}")
    finally:
        db.close()
        scan_status.clear_status()


def scan_one_account(email: str):
    """One-off scan for a single account (triggered right after login)."""
    db = SessionLocal()
    try:
        account = db.query(UserAccount).filter(UserAccount.email == email).first()
        if account is None:
            return
        try:
            stats = scan_account(db, account)
            print(f"[scan:onboard] {account.email}: {stats}")
        except Exception as e:
            db.rollback()
            print(f"[scan:onboard] error for {account.email}: {e}")
    finally:
        db.close()
        scan_status.clear_status()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler.add_job(
        scan_all_accounts,
        "interval",
        minutes=SCAN_INTERVAL_MINUTES,
        id="scan_all_accounts",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, max_age=86400)
templates = Jinja2Templates(directory="templates")


def _mail_url(message_ref: str) -> str:
    """Build a macOS Mail deep link (message:// scheme) from an RFC-822 Message-ID.

    Opens the message directly in Mail.app when the account is configured there.
    """
    ref = (message_ref or "").strip().strip("<>")
    if not ref:
        return ""
    return "message://%3C" + urllib.parse.quote(ref, safe="") + "%3E"


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

    creds_dict = {
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

    # Persist credentials so the background scanner can run without a session,
    # then kick off an immediate scan so a new user sees results quickly.
    email = user_info.get("email")
    if email:
        db = SessionLocal()
        try:
            upsert_account(
                db,
                email=email,
                credentials=creds_dict,
                name=user_info.get("name"),
                picture=user_info.get("picture"),
            )
        finally:
            db.close()
        scheduler.add_job(scan_one_account, args=[email], id=f"onboard:{email}",
                          replace_existing=True)

    return RedirectResponse("/")


@app.post("/job/{job_id}/edit")
async def edit_job(
    job_id: int,
    request: Request,
    job_title: str = Form(""),
    status: str = Form(""),
):
    """Manually edit a job's title/status. Recorded as a manual entry in the
    job's email timeline; a later real email can still override the status."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/auth/login", status_code=303)

    db = SessionLocal()
    try:
        job = (
            db.query(JobApplication)
            .filter(JobApplication.id == job_id, JobApplication.user_email == user["email"])
            .first()
        )
        if job is None:
            return RedirectResponse("/jobs", status_code=303)

        old_title, old_status = job.job_title, job.status
        new_title = job_title.strip() or job.job_title
        new_status = status.strip() or job.status
        job.job_title = new_title
        job.status = new_status
        job.updated_at = datetime.utcnow()
        job.unread = False  # user made this change, so it's already "seen"

        changes = []
        if new_status != old_status:
            changes.append(f"status to {new_status}")
        if new_title != old_title:
            changes.append(f"role to {new_title}")
        summary = (
            "You updated the " + " and ".join(changes) + "."
            if changes else "You saved the details."
        )

        db.add(JobEmail(
            job_id=job.id,
            user_email=user["email"],
            email_id=f"manual-{job.id}-{secrets.token_hex(6)}",
            message_ref="",
            subject="Manual update",
            paraphrase=summary,
            received_at=datetime.utcnow(),
        ))
        db.commit()
        scan_status.bump_revision()
    finally:
        db.close()

    return RedirectResponse("/jobs", status_code=303)


@app.post("/job/{job_id}/seen")
async def mark_seen(job_id: int, request: Request):
    """Clear a job's unread flag once the user opens it (removes the bell)."""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"ok": False}, status_code=401)
    db = SessionLocal()
    try:
        job = (
            db.query(JobApplication)
            .filter(JobApplication.id == job_id, JobApplication.user_email == user["email"])
            .first()
        )
        if job is not None and job.unread:
            job.unread = False
            db.commit()
    finally:
        db.close()
    return JSONResponse({"ok": True})


@app.get("/scan-status")
async def scan_status_endpoint():
    """Current background-parser activity, polled by the dashboard."""
    return JSONResponse(scan_status.get_status())


@app.get("/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")


@app.post("/reset")
async def reset(request: Request):
    """Wipe this user's parsed data + dedup ledger and reparse the inbox from scratch."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/auth/login", status_code=303)

    email = user["email"]
    db = SessionLocal()
    try:
        db.query(JobEmail).filter(JobEmail.user_email == email).delete()
        db.query(JobApplication).filter(JobApplication.user_email == email).delete()
        db.query(ProcessedEmail).filter(ProcessedEmail.user_email == email).delete()

        account = db.query(UserAccount).filter(UserAccount.email == email).first()
        can_rescan = bool(account and account.refresh_token)
        if account:
            account.last_scanned_at = None  # force a full 30-day re-onboard
        db.commit()
    finally:
        db.close()

    # Reparse in the background so the redirect returns immediately.
    if can_rescan:
        scheduler.add_job(scan_one_account, args=[email], id=f"reset:{email}",
                          replace_existing=True)

    return RedirectResponse("/jobs", status_code=303)


@app.get("/jobs", response_class=HTMLResponse)
async def jobs(request: Request):
    """Render stored job applications. Pure DB read — scanning happens in the
    background job, never in the request path."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse("/auth/login")

    db = SessionLocal()
    try:
        saved_jobs = (
            db.query(JobApplication)
            .filter(JobApplication.user_email == user["email"])
            .order_by(JobApplication.updated_at.desc())
            .all()
        )
        jobs_list = []
        for job in saved_jobs:
            emails = (
                db.query(JobEmail)
                .filter(JobEmail.job_id == job.id)
                .order_by(JobEmail.received_at.desc())
                .all()
            )
            jobs_list.append({
                "id": job.id,
                "company": job.company,
                "job_title": job.job_title,
                "status": job.status,
                "unread": bool(job.unread),
                "date": (job.applied_date or job.created_at).strftime("%d-%m-%Y")
                if (job.applied_date or job.created_at)
                else "",
                "emails": [
                    {
                        "subject": e.subject or "(no subject)",
                        "paraphrase": e.paraphrase or "",
                        "date": e.received_at.strftime("%d-%m-%Y") if e.received_at else "",
                        "mail_url": _mail_url(e.message_ref),
                    }
                    for e in emails
                ],
            })
    finally:
        db.close()

    return templates.TemplateResponse(
        request=request,
        name="jobs.html",
        context={
            "request": request,
            "user": user,
            "jobs": jobs_list,
            "scan_revision": scan_status.get_status().get("revision", 0),
        },
    )
