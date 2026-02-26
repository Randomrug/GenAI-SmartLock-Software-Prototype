#!/usr/bin/env python3
"""
Smart Lock System - Initial Setup & Diagnostics
Helps with face enrollment, voice enrollment, and system configuration
"""
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*70)
print("[LOCK] GenAI SMART LOCK SYSTEM - SETUP & DIAGNOSTICS")
print("="*70 + "\n")

# Check .env file
print("[INFO] CONFIGURATION CHECK")
print("-" * 70)

env_file = '.env'
if os.path.exists(env_file):
    print(f"[OK] .env file found at: {os.path.abspath(env_file)}")
    with open(env_file, 'r') as f:
        content = f.read()
        if 'OPENROUTER_API_KEY=' in content:
            print("[OK] OPENROUTER_API_KEY is configured")
        else:
            print("[ERROR] OPENROUTER_API_KEY is NOT configured")
else:
    print(f"[ERROR] .env file NOT found")
    print(f"   Expected at: {os.path.abspath(env_file)}")
    print(f"   Run: copy .env.example .env")

# Check database
print("\n[DATA] DATABASE CHECK")
print("-" * 70)

db_path = 'smart_lock_events.db'
if os.path.exists(db_path):
    size_mb = os.path.getsize(db_path) / (1024*1024)
    print(f"[OK] Database found: {db_path} ({size_mb:.2f} MB)")
else:
    print(f"[WARNING]  Database not found: {db_path}")
    print(f"   Database will be created on first run")

# Check face database
print("\n[USER] FACE RECOGNITION CHECK")
print("-" * 70)

face_db_path = 'embeddings/face_db.pkl'
if os.path.exists(face_db_path):
    print(f"[OK] Face database found: {face_db_path}")
    # Count faces
    import pickle
    try:
        with open(face_db_path, 'rb') as f:
            face_db = pickle.load(f)
        print(f"   Registered faces: {list(face_db.keys())}")
    except Exception as e:
        print(f"   Error reading face database: {e}")
else:
    print(f"[WARNING]  Face database NOT found: {face_db_path}")
    print(f"   Action: Follow the face enrollment process below")

# Check voice template
print("\n[VOICE] VOICE RECOGNITION CHECK")
print("-" * 70)

voice_template_path = 'VoiceRecognition/templates/my_voice_template.pt'
if os.path.exists(voice_template_path):
    print(f"[OK] Voice template found: {voice_template_path}")
else:
    print(f"[WARNING]  Voice template NOT found: {voice_template_path}")
    print(f"   Action: Follow the voice enrollment process below")

# Check pretrained models
print("\n[AI] PRETRAINED MODELS CHECK")
print("-" * 70)

models_dir = 'VoiceRecognition/pretrained_models/spkrec-ecapa-voxceleb'
if os.path.exists(models_dir):
    files = os.listdir(models_dir)
    print(f"[OK] SpeechBrain models found in: {models_dir}")
    print(f"   Files: {', '.join(files)}")
else:
    print(f"[WARNING]  Models directory NOT found: {models_dir}")
    print(f"   Note: Models will be downloaded on first voice enrollment")

# Check FaceNet models
print("\n📷 FACENET MODELS CHECK")
print("-" * 70)

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    print("[OK] FaceNet models are available")
    print("   MTCNN (Face Detection): Ready")
    print("   InceptionResnetV1 (Face Embedding): Ready")
except ImportError:
    print("[ERROR] FaceNet models NOT available")
    print("   Run: pip install facenet-pytorch")

# Check OpenRouter API
print("\n[WEB] OPENROUTER API CHECK")
print("-" * 70)

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENROUTER_API_KEY')
if api_key and api_key.startswith('sk-or-v1'):
    print(f"[OK] OpenRouter API key is configured")
    print(f"   Key: {api_key[:20]}...{api_key[-20:]}")
else:
    print(f"[ERROR] OpenRouter API key NOT configured correctly")
    print(f"   Make sure .env has OPENROUTER_API_KEY=sk-or-v1-...")

# Show next steps
print("\n" + "="*70)
print("[NOTE] SETUP INSTRUCTIONS")
print("="*70 + "\n")

print("1️⃣  CONFIGURE ENVIRONMENT VARIABLES")
print("   ✓ Copy .env.example to .env (already done)")
print("   ✓ Add your OpenRouter API key to .env")
print("   ✓ API key should start with: sk-or-v1-\n")

print("2️⃣  ENROLL FACE (Optional but recommended)")
print("   ✓ Create folder: FaceNet/enroll/your_name/")
print("   ✓ Add 3-5 clear face photos to that folder")
print("   ✓ Run: python FaceNet/enroll_faces.py")
print("   ✓ This creates: embeddings/face_db.pkl\n")

print("3️⃣  ENROLL VOICE (Optional but recommended)")
print("   ✓ Run: python VoiceRecognition/enroll_my_voice.py")
print("   ✓ Follow the prompts to record 5 voice samples")
print("   ✓ This creates: VoiceRecognition/templates/my_voice_template.pt\n")

print("4️⃣  START THE SYSTEM")
print("   ✓ Run: python app.py")
print("   ✓ Or: python run.py")
print("   ✓ Access web interface at: http://localhost:5000\n")

print("5️⃣  TEST AUTHENTICATION")
print("   ✓ PIN: 1234 (default, edit in pin.txt)")
print("   ✓ Try authentication through web interface")
print("   ✓ Check logs in: logs/ directory\n")

# Option to proceed with enrollment
print("="*70)
print("[TARGET] QUICK START")
print("="*70 + "\n")

choice = input("Would you like to:\n[1] Continue with setup\n[2] Start Flask app\n[3] Exit\nChoice (1-3): ").strip()

if choice == '1':
    print("\n[CONFIG] Running setup...\n")
    # You could add enrollment scripts here
    print("To enroll face: python FaceNet/enroll_faces.py")
    print("To enroll voice: python VoiceRecognition/enroll_my_voice.py")
elif choice == '2':
    print("\n[START] Starting Flask app...\n")
    os.system('python app.py')
else:
    print("\n👋 Goodbye!")

print("\n" + "="*70)
