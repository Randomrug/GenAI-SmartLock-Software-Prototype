import requests
import os
import time
import pyttsx3
from .send_email import send_email

BOT_TOKEN = "8261455473:AAFpnEj7ZA2A2Uq7idHpSMpOyIzqFktu8AY"
CHAT_ID = 5020315161

def make_call():
    msg = "SMARTLOCK ALERT: Multiple unusual login attempts detected. System temporary locked for security. Voice message: "

    # Telegram emergency
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    r = requests.post(url, data=payload)

    if r.status_code == 200:
        print("[OK] Telegram emergency sent!")
    else:
        print("[ERROR] Telegram failed:", r.text)

    # Email emergency
    send_email("SMARTLOCK UNUSUAL LOGIN ALERT", "Multiple unusual login attempts have triggered system lockout. Review the security logs to confirm this was authorized access.")

    # Call simulation (ringing)
    print("\n📞 Incoming Call Simulation...\n")
    for i in range(8):
        os.system("echo \a")  # Windows beep
        print(f"📞 Ringing... {i+1}")
        time.sleep(1)

    # Voice message
    engine = pyttsx3.init()
    engine.say("Alert. Smart lock system has been locked due to repeated unusual login attempts. If this was not you, please contact security immediately.")
    engine.say("Immediate attention required.")
    engine.runAndWait()

    print("[OK] Call simulation completed!")


