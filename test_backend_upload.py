import requests

url = "http://localhost:5000/api/authenticate"

data = {
    "action": "IN",
    "manual_date": "2026-02-19",
    "manual_time": "10:45",
    "pin": "1234"
}

# Use a real image and audio file if available. Here, asset.enc is used as a placeholder for both.
files = {
    "face_image": ("face.jpg", open("assets/asset.enc", "rb"), "image/jpeg"),
    "voice_audio": ("voice.webm", open("assets/asset.enc", "rb"), "audio/webm")
}

response = requests.post(url, data=data, files=files)
print("Status Code:", response.status_code)
print("Response:", response.text)
