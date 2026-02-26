Here’s a clean, friendly, professional README that explains everything properly — without turning into a scrolling marathon and without sounding like a stand-up comedy show.

---

# 🔐 GenAI Smart Lock – Multi-Factor Adaptive Security System

An intelligent physical access control system that combines **Face Recognition, Voice Recognition, PIN Authentication, Behavioral Analysis, and AI-Driven Anomaly Detection** into one adaptive security pipeline.

This isn’t just “face + PIN = open door.”
This system *analyzes context, behavior, patterns, and risk* before deciding whether your door should trust you today.

---

## 🧠 What This System Does

At a high level, the system:

* Registers users with **face + voice + PIN**
* Authenticates using live biometric capture
* Scores user behavior patterns
* Fuses all scores into one decision
* Uses GenAI for anomaly interpretation
* Triggers alerts and lockouts if needed
* Logs everything into a monitoring dashboard

The complete authentication logic is described in the Authentication Pipeline Architecture  and the data movement between modules is documented in the Data Flow Pipeline .

---

# 🧩 Core Modules (Short & Clear)

Architecture reference: Module-wise Architecture Overview 

### 1️⃣ Face Recognition (FaceNet)

* Captures live image from camera
* Converts face into embeddings
* Compares with stored embeddings
* Triggers anomaly if mismatch

Files:

```
FaceNet/capture_live.py  
FaceNet/verify_face.py  
FaceNet/enroll_faces.py  
```

Used during:

* Registration (store embeddings)
* Authentication (verify identity)

---

### 2️⃣ Voice Recognition

* Records user voice sample
* Extracts voice embeddings
* Compares against enrolled voice
* Optional anti-spoof handling

Files:

```
VoiceRecognition/enroll_my_voice.py  
VoiceRecognition/smart_lock.py  
```

Works alongside face module for multi-factor authentication.

---

### 3️⃣ PIN Security (SHA-256 Hashing)

* User sets PIN
* PIN is hashed (never stored as plaintext)
* Authentication hashes input and compares

Files:

```
Configuration/pin_security.py  
pin_hash.json  
```

Supports:

* PIN reset
* Backup & recovery flows

---

### 4️⃣ Behavior Model

* Monitors access timing
* Frequency patterns
* Usage history
* Detects anomalies using decision trees + statistical detection

Files:

```
models/behavior_model.py  
models/anomaly_detector.py  
```

Feeds behavioral score into fusion engine.

---

### 5️⃣ Score Fusion Module

* Combines:

  * Face score
  * Voice score
  * PIN score
  * Behavior score
* Applies weighted logic
* Produces final authentication score

File:

```
models/score_fusion.py
```

This is the **central decision engine**.

---

### 6️⃣ AI Safety & GenAI Module

* Analyzes anomaly patterns
* Generates recommendations
* Adjusts risk thresholds dynamically

Files:

```
ai_safety.py  
Model/explanation_model.py  
```

This layer adds contextual intelligence beyond simple rule checking.

---

### 7️⃣ Alert System

Sends alerts when:

* Repeated failures occur
* High anomaly detected
* Lockout triggered

Channels:

* Email
* SMS
* Telegram Bot

Files:

```
Alert_system/alert.py  
Alert_system/send_email.py  
Alert_system/telegram_webhook.py  
```

Evidence (face image, voice sample, logs) is attached.

---

### 8️⃣ Lockout & Reset Module

Handles:

* Automatic lockout after repeated failures
* Vacation mode (disable biometric access)
* Admin recovery reset
* Log reset & DB reset

Files:

```
reset_lockout.py  
reset_database.py  
reset_access_log.py  
```

---

### 9️⃣ Dashboard & Monitoring

* Real-time system status
* Access logs
* Anomaly reports
* Analytics

Files:

```
templates/index.html  
logs/  
```

Provides centralized visibility into the entire system.

---

# 🔄 Full Integration Flow

Technical background: Detailed Description 

## Registration Phase

1. User enrolls:

   * Face
   * Voice
   * PIN
2. Face & voice embeddings stored
3. PIN hashed and stored securely
4. Behavior baseline initialized

---

## Authentication Phase

1. Live face captured
2. Live voice captured
3. PIN entered
4. Each module verifies independently
5. Behavior model scores attempt
6. Scores fused into final authentication score
7. GenAI analyzes anomaly patterns
8. Access decision made

If:

* ✅ Normal → Access granted
* ⚠ Mild anomaly → Additional checks
* ❌ High anomaly → Access denied + alerts + lockout

---

## Database Interaction

During authentication:

* Face & voice embeddings retrieved
* PIN hash retrieved
* Behavior logs accessed
* Attempt logged
* Dashboard updated

Everything flows through the pipeline described in .

---

# ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Randomrug/GenAI-SmartLock-Software-Prototype.git
cd GenAI-SmartLock-Software-Prototype
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Configure Environment

* Copy `.env.example`
* Rename to `.env`
* Add:

  * Email credentials
  * Telegram bot token
  * SMS API key
  * GenAI/OpenRouter key (optional for simulation)

---

# ▶️ Running the System

### Step 1: Register Owner

Enroll face:

```bash
python FaceNet/enroll_faces.py
```

Enroll voice:

```bash
python VoiceRecognition/enroll_my_voice.py
```

Set PIN:

```bash
python Configuration/pin_security.py
```

This creates:

* Face embeddings
* Voice embeddings
* Hashed PIN entry
* Database records

---

### Step 2: Start Main System

```bash
python run.py
```

---

### Step 3: Access Dashboard

Open browser:

```
http://localhost:5000
```

You can:

* Monitor logs
* View anomaly scores
* Track authentication attempts
* Observe alerts

---

# 🧪 Testing

```bash
python -m unittest discover TEST
```

---

# 📌 System Highlights

* Multi-factor authentication (Face + Voice + PIN)
* Behavioral anomaly detection
* AI-driven adaptive scoring
* Multi-channel alert system
* Lockout & vacation mode
* Real-time dashboard monitoring
* Modular, extensible architecture

---

Heyyy guyssss 👋
Any queries, collaborations, or suggestions — feel free to ping me at **[rithikaarulmozhi21@gmail.com](mailto:rithikaarulmozhi21@gmail.com)**

— **Randomrug** 🚀
