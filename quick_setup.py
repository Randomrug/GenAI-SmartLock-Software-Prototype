#!/usr/bin/env python3
"""
Quick Test Setup - Creates test data for demonstration
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*70)
print("[LOCK] SMART LOCK SYSTEM - QUICK TEST SETUP")
print("="*70 + "\n")

# Create face embedding directory
print("[FOLDER] Creating directory structure...")
os.makedirs('embeddings', exist_ok=True)
os.makedirs('VoiceRecognition/templates', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Create test face database with simulated embeddings
print("[USER] Creating test face database...")
face_db = {
    'rithika': np.random.randn(512).astype(np.float32),  # Simulated face embedding
    'admin': np.random.randn(512).astype(np.float32),     # Another test user
}

# Normalize embeddings
for name in face_db:
    face_db[name] = face_db[name] / np.linalg.norm(face_db[name])

# Save face database
face_db_path = 'embeddings/face_db.pkl'
with open(face_db_path, 'wb') as f:
    pickle.dump(face_db, f)

print(f"[OK] Face database created: {face_db_path}")
print(f"   Registered users: {list(face_db.keys())}")

# Create test voice template
print("\n[VOICE] Creating test voice template...")
import torch

voice_template = torch.randn(192, dtype=torch.float32)
voice_template = voice_template / torch.norm(voice_template)

voice_template_path = 'VoiceRecognition/templates/my_voice_template.pt'
torch.save(voice_template, voice_template_path)

print(f"[OK] Voice template created: {voice_template_path}")

# Create test PIN
print("\nPIN Configuration...")
pin_file = 'pin.txt'
if not os.path.exists(pin_file):
    with open(pin_file, 'w') as f:
        f.write('1234')
    print(f"[OK] PIN file created: {pin_file}")
    print(f"   Default PIN: 1234")
else:
    with open(pin_file, 'r') as f:
        pin = f.read().strip()
    print(f"[OK] PIN file exists: {pin_file}")
    print(f"   Current PIN: {pin}")

print("\n" + "="*70)
print("[OK] QUICK SETUP COMPLETE!")
print("="*70)

print("\n[NOTE] NEXT STEPS:")
print("1. Check .env file has OPENROUTER_API_KEY configured")
print("2. Run: python app.py")
print("3. Visit: http://localhost:5000")
print("4. Test authentication with:")
print("   - Action: UNLOCK")
print("   - PIN: 1234")
print("   - Optional: Upload face/voice (simulated mode)")
print("\n")
