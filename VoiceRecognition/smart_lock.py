# smart_lock.py - Multi-Owner Voice Recognition
"""
Voice authentication for multiple owners
Matches voice against all enrolled owner templates
"""
import sounddevice as sd
import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
import os
import time
import sys
import pickle

# Global variables for multi-owner support
verification = None
voice_templates = {}  # {"rithika": tensor, "sid": tensor, ...}
VOICE_THRESHOLD = 0.65

def init_voice_system():
    """Initialize voice recognition system with all owner templates"""
    global verification, voice_templates
    
    print("🔊 Initializing Multi-Owner Voice System...\n")
    
    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        local_strategy=LocalStrategy.COPY
    )
    print("[OK] Model loaded!\n")
    
    # Load all voice templates from templates directory
    templates_dir = "VoiceRecognition/templates"
    if not os.path.exists(templates_dir):
        templates_dir = "templates"
    
    if not os.path.exists(templates_dir):
        print("[ERROR] No templates directory found!")
        return False
    
    voice_templates = {}
    template_files = [f for f in os.listdir(templates_dir) if f.endswith("_voice_template.pt")]
    
    if not template_files:
        print(f"[WARNING]  No voice templates found in {templates_dir}")
        print("   Please run: python VoiceRecognition/enroll_my_voice.py")
        return False
    
    for template_file in sorted(template_files):
        owner_name = template_file.replace("_voice_template.pt", "")
        template_path = os.path.join(templates_dir, template_file)
        try:
            voice_templates[owner_name] = torch.load(template_path)
            print(f"[OK] Loaded template: {owner_name}")
        except Exception as e:
            print(f"[ERROR] Failed to load {owner_name}: {e}")
    
    print(f"\n[SECURE] System armed with {len(voice_templates)} registered owner(s)\n")
    return len(voice_templates) > 0

def verify_voice(waveform):
    """
    Verify voice against all enrolled owner templates
    Returns: (owner_name, best_score, is_match)
    """
    if not voice_templates:
        return "Unknown", 0.0, False
    
    # Get embedding from live audio
    emb_live = verification.encode_batch(waveform)
    
    # Handle tensor dimensions
    if emb_live.dim() == 3:
        emb_live = emb_live.mean(dim=1)
    elif emb_live.dim() == 1:
        emb_live = emb_live.unsqueeze(0)
    
    # Compare against all owner templates
    scores = {}
    for owner_name, template in voice_templates.items():
        emb_template = template
        if emb_template.dim() == 1:
            emb_template = emb_template.unsqueeze(0)
        
        # Calculate cosine similarity
        score = torch.nn.functional.cosine_similarity(
            emb_template.view(1, -1),
            emb_live.view(1, -1),
            dim=1
        )
        scores[owner_name] = float(score[0].item())
    
    # Find best match
    best_owner = max(scores, key=scores.get)
    best_score = scores[best_owner]
    
    print(f"\n[SEARCH] Voice Verification Results:")
    print("=" * 50)
    for owner, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        status = "[OK]" if owner == best_owner else "  "
        match_status = "MATCH" if score >= VOICE_THRESHOLD else "no match"
        print(f"{status} {owner:15} | Score: {score:.4f} [{match_status}]")
    print("=" * 50)
    
    is_match = best_score >= VOICE_THRESHOLD
    
    if is_match:
        print(f"\n[OK] VOICE VERIFIED: {best_owner} (Score: {best_score:.4f})\n")
    else:
        print(f"\n[ERROR] NO MATCH: Best match is {best_owner} ({best_score:.4f}), threshold: {VOICE_THRESHOLD}\n")
    
    return best_owner, best_score, is_match

SAMPLE_RATE = 16000
DURATION = 5

def record_live():
    """Record live voice sample"""
    print("[VOICE] Listening... Speak your phrase now!")
    print("   (Recording for 5 seconds...)\n")
    time.sleep(0.5)
    
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    
    # CORRECT: [1, time]
    waveform = torch.from_numpy(audio.flatten())  # Flatten to 1D [time]
    waveform = waveform.unsqueeze(0)              # Add batch → [1, time]
    return waveform

# CLI code - only runs if this file is executed directly
if __name__ == '__main__':
    if not init_voice_system():
        print("[ERROR] Failed to initialize voice system")
        exit(1)
    
    print("=" * 60)
    print("[LOCK] MULTI-OWNER VOICE AUTHENTICATION")
    print("=" * 60)
    print(f"Registered owners: {', '.join(sorted(voice_templates.keys()))}")
    print("=" * 60)
    print("\nPress ENTER to attempt voice authentication (Ctrl+C to quit)\n")
    
    # Allow up to 3 attempts; exit on success or after 3 failures
    MAX_ATTEMPTS = 3
    attempts = 0
    
    while True:
        input()
        waveform = record_live()
        
        # Verify voice against all owners
        detected_owner, voice_score, voice_match = verify_voice(waveform)
        
        if voice_match:
            print(f"🎉 ACCESS GRANTED! Welcome, {detected_owner.upper()}!\n")
            break
        else:
            attempts += 1
            tries_left = MAX_ATTEMPTS - attempts
            
            if attempts >= MAX_ATTEMPTS:
                print(f"WARNING: {MAX_ATTEMPTS} failed attempts. System locked.\n")
                exit(1)
            
            print(f"⏳ Tries remaining: {tries_left}\n")