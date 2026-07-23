"""Dev helper: seed the DB with fake job applications + email threads for UI.

Run:  python seed_db.py [user_email]

Idempotent — wipes this user's job_applications and job_emails first, then
reinserts the fixture set. Leaves user_accounts and processed_emails alone.
"""
import sys
from datetime import datetime, timedelta

from database import init_db, SessionLocal, JobApplication, JobEmail

SEED_USER = sys.argv[1] if len(sys.argv) > 1 else "jones.oscar.pro@gmail.com"

NOW = datetime(2026, 7, 23, 9, 0, 0)

# (company, title, status, hours_since_update, [ (subject, paraphrase, days_ago), ... ])
# Emails listed oldest -> newest; the job's "applied" date is the oldest one.
FIXTURES = [
    ("Monzo", "Backend Engineer", "Offer", 2, [
        ("Your application to Monzo", "They confirmed they received your application.", 18),
        ("Technical interview scheduled", "They booked you in for a technical interview.", 10),
        ("Offer — Backend Engineer", "They offered you the role and shared next steps.", 5),
    ]),
    ("Notion", "Product Designer", "Offer", 6, [
        ("Application received", "They acknowledged your application.", 24),
        ("Portfolio review", "They liked your portfolio and moved you to the next round.", 14),
        ("We'd love to offer you the role", "They extended a formal offer.", 9),
    ]),
    ("Stripe", "Senior Backend Engineer", "Interview", 20, [
        ("Thanks for applying to Stripe", "They confirmed receipt of your application.", 16),
        ("First-round interview", "They invited you to a first-round interview.", 7),
    ]),
    ("Linear", "Full Stack Engineer", "Interview", 30, [
        ("Application received", "They received your application.", 20),
        ("Let's chat", "They invited you to an intro call with the team.", 11),
    ]),
    ("Anthropic", "Research Engineer", "Assessment", 44, [
        ("Your Anthropic application", "They acknowledged your application.", 15),
        ("Take-home assessment", "They sent a take-home assessment to complete.", 6),
    ]),
    ("OpenAI", "Member of Technical Staff", "Assessment", 52, [
        ("Application received", "They confirmed your application was received.", 20),
        ("Coding assessment", "They asked you to complete an online coding assessment.", 12),
    ]),
    ("Datadog", "Site Reliability Engineer", "Awaiting Response", 70, [
        ("We received your application", "They confirmed they got your application.", 3),
    ]),
    ("Ramp", "Software Engineer, Backend", "Awaiting Response", 74, [
        ("Thanks for applying to Ramp", "They acknowledged your application.", 2),
    ]),
    ("Retool", "Product Engineer", "Awaiting Response", 96, [
        ("Application received", "They confirmed receipt of your application.", 8),
    ]),
    ("Vercel", "Developer Advocate", "Other", 120, [
        ("Thanks for your interest", "They added you to their talent pool for future roles.", 14),
    ]),
    ("Figma", "Frontend Engineer", "Rejected", 150, [
        ("Application received", "They acknowledged your application.", 22),
        ("Update on your application", "They decided not to move forward this time.", 18),
    ]),
    ("Airtable", "Engineering Manager", "Rejected", 180, [
        ("Your Airtable application", "They received your application.", 28),
        ("Application update", "They passed on your application for now.", 21),
    ]),
]


def main():
    init_db()
    db = SessionLocal()
    try:
        del_e = db.query(JobEmail).filter(JobEmail.user_email == SEED_USER).delete()
        del_j = db.query(JobApplication).filter(JobApplication.user_email == SEED_USER).delete()

        n_emails = 0
        for i, (company, title, status, updated_hours, emails) in enumerate(FIXTURES):
            applied = NOW - timedelta(days=max(d for _, _, d in emails))
            job = JobApplication(
                user_email=SEED_USER,
                company=company,
                job_title=title,
                status=status,
                email_id=f"seed-{i}",
                email_subject=emails[-1][0],
                email_body=emails[-1][1],
                applied_date=applied,
                created_at=applied,
                updated_at=NOW - timedelta(hours=updated_hours),
            )
            db.add(job)
            db.flush()  # get job.id
            for j, (subject, paraphrase, days_ago) in enumerate(emails):
                db.add(
                    JobEmail(
                        job_id=job.id,
                        user_email=SEED_USER,
                        email_id=f"seed-{i}-{j}",
                        message_ref=f"seed-{i}-{j}@mail.example.com",
                        subject=subject,
                        paraphrase=paraphrase,
                        received_at=NOW - timedelta(days=days_ago),
                    )
                )
                n_emails += 1
        db.commit()
        print(f"Seeded {len(FIXTURES)} applications and {n_emails} emails for {SEED_USER} "
              f"(removed {del_j} jobs / {del_e} emails).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
