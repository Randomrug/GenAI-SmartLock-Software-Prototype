import requests
import os
from .send_email import send_email

BOT_TOKEN = "8261455473:AAFpnEj7ZA2A2Uq7idHpSMpOyIzqFktu8AY"
CHAT_ID = 5020315161

def send_telegram(message):
    """Send a Telegram alert (real bot integration)"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    r = requests.post(url, data=payload)
    if r.status_code == 200:
        print("[OK] Telegram mild alert sent!")
    else:
        print("[ERROR] Telegram mild alert failed:", r.text)
import requests
import os
from .send_email import send_email

BOT_TOKEN = "8261455473:AAFpnEj7ZA2A2Uq7idHpSMpOyIzqFktu8AY"
CHAT_ID = 5020315161

def send_sms(image_paths=None):
    """
    Send SMS warning with optional images for investigation
    
    Args:
        image_paths: List of image file paths to send along with the alert
    """
    # Telegram message
    msg = "[ALERT] SmartLock: Unusual login attempts detected - biometric verification failed. Please check if this was you."

    # Send initial text message
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg}
    r = requests.post(url, data=payload)

    if r.status_code == 200:
        print("[OK] Telegram warning sent!")
    else:
        print("[ERROR] Telegram failed:", r.text)

    # Send captured images if available
    if image_paths:
        send_investigation_images(image_paths, "Unusual Login Attempt - Investigation Images")
    
    # Email alert
    send_email("SmartLock Unusual Login Alert [ALERT]", msg)

def send_investigation_images(image_paths, caption="Investigation Images"):
    """Send captured images to Telegram for investigation"""
    if not image_paths:
        return
    
    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    import json
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMediaGroup"
    media = []
    valid_images = []
    for i, img_path in enumerate(image_paths):
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            valid_images.append(img_path)
            media_item = {
                "type": "photo",
                "media": f"attach://photo{i}",
                "caption": caption if i == 0 else ""
            }
            media.append(media_item)
    if not valid_images:
        print("[WARNING]  No valid images to send")
        return
    try:
        files = {}
        for i, img_path in enumerate(valid_images):
            files[f'photo{i}'] = open(img_path, 'rb')
        payload = {
            "chat_id": CHAT_ID,
            "media": json.dumps(media)
        }
        r = requests.post(url, data=payload, files=files)
        for file_obj in files.values():
            file_obj.close()
        if r.status_code == 200:
            print(f"[OK] Sent {len(valid_images)} investigation image(s) to Telegram!")
        else:
            print(f"[ERROR] Failed to send images: {r.text}")
    except Exception as e:
        print(f"[ERROR] Error sending images: {e}")

