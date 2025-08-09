# src/ml/email_zip_sender.py

import smtplib
from email.message import EmailMessage
from pathlib import Path


def send_zip_email(
    zip_path: Path,
    to_email: str,
    from_email: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
):
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    msg = EmailMessage()
    msg["Subject"] = "📦 Annotated Fintech Bundle"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(
        "Attached is your annotated fintech code analysis bundle (ZIP).\n\nThanks,\nCode Analyser"
    )

    with open(zip_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="zip", filename=zip_path.name)

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
        smtp.login(smtp_user, smtp_pass)
        smtp.send_message(msg)

    print(f"✅ Email sent to: {to_email}")
