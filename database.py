import os
import json
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./jobe.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class JobApplication(Base):
    __tablename__ = "job_applications"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    company = Column(String, index=True)
    job_title = Column(String)
    status = Column(String)
    email_id = Column(String, unique=True)
    email_subject = Column(String)
    email_body = Column(Text, nullable=True)
    applied_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserAccount(Base):
    """A signed-in user's persisted OAuth credentials and scan state.

    The background scanner runs with no HTTP session, so it reads credentials
    from here rather than the session cookie. ``last_scanned_at`` is ``None``
    until the first (onboarding) scan completes.
    """

    __tablename__ = "user_accounts"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    picture = Column(String, nullable=True)
    token = Column(String, nullable=True)
    refresh_token = Column(String, nullable=True)
    token_uri = Column(String, nullable=True)
    client_id = Column(String, nullable=True)
    client_secret = Column(String, nullable=True)
    scopes = Column(Text, nullable=True)  # JSON-encoded list
    last_scanned_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProcessedEmail(Base):
    """One row per email ever handed to the parser.

    This is the duplicate-parse guard: before calling Gemini we check whether
    an ``email_id`` already exists here and skip it if so, guaranteeing each
    email is graded at most once regardless of scan-window overlap or restarts.
    """

    __tablename__ = "processed_emails"

    id = Column(Integer, primary_key=True, index=True)
    email_id = Column(String, unique=True, index=True)
    user_email = Column(String, index=True)
    internal_date = Column(String, nullable=True)
    is_job_application = Column(Boolean, default=False)
    processed_at = Column(DateTime, default=datetime.utcnow)


class JobEmail(Base):
    """One row per job-related email, linked to its JobApplication.

    A single application collapses many emails (application received, assessment,
    interview, offer …); this table keeps each one so a job row can expand to
    show its history. ``paraphrase`` is the model's one-sentence summary.
    """

    __tablename__ = "job_emails"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("job_applications.id"), index=True)
    user_email = Column(String, index=True)
    email_id = Column(String, unique=True)
    message_ref = Column(String, nullable=True)  # RFC-822 Message-ID, for message:// links
    subject = Column(String)
    paraphrase = Column(Text, nullable=True)
    received_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


def upsert_account(db: Session, *, email: str, credentials: dict, name: str | None = None,
                   picture: str | None = None) -> UserAccount:
    """Create or update a user's persisted account/credentials.

    ``credentials`` is the same dict shape stored in the session (see
    ``main.py``). A refresh token is only returned by Google on the first
    consent, so an absent/empty refresh_token never overwrites a stored one.
    """
    account = db.query(UserAccount).filter(UserAccount.email == email).first()
    if account is None:
        account = UserAccount(email=email)
        db.add(account)

    account.name = name if name is not None else account.name
    account.picture = picture if picture is not None else account.picture
    account.token = credentials.get("token")
    if credentials.get("refresh_token"):
        account.refresh_token = credentials.get("refresh_token")
    account.token_uri = credentials.get("token_uri")
    account.client_id = credentials.get("client_id")
    account.client_secret = credentials.get("client_secret")
    account.scopes = json.dumps(credentials.get("scopes") or [])

    db.commit()
    db.refresh(account)
    return account
