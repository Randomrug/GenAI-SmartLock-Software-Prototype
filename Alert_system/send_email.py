import smtplib
from email.mime.text import MIMEText

SENDER_EMAIL = "genai.alert.system@gmail.com"
APP_PASSWORD = "xhkt ijdl gllh tdwj"
RECEIVER_EMAIL = "rithikaarulmozhi21@gmail.com"   # default receiver (can be overridden)

def send_email(subject, body, to_email: str = None):
    """Send email using configured SMTP credentials.

    Args:
        subject: Email subject
        body: Email body
        to_email: Optional recipient email. If not provided, uses default RECEIVER_EMAIL
    """
    recipient = to_email or RECEIVER_EMAIL

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(SENDER_EMAIL, APP_PASSWORD)
    server.sendmail(SENDER_EMAIL, recipient, msg.as_string())
    server.quit()

    print(f"[OK] Email sent to {recipient}!")
