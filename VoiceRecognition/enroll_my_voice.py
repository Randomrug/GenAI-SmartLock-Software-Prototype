# enroll_my_voice.py - Multi-Owner Voice Enrollment
"""
Enroll voice for a specific owner
Saves as: templates/{owner_name}_voice_template.pt
"""
import sounddevice as sd
import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
import os
import time

SAMPLE_RATE = 16000
DURATION = 5

def record(sample_num):
    """Record voice sample"""
    print(f"   [VOICE] Sample {sample_num}/5 - Say clearly: 'Hey lock, open the door please'")
    print("   Recording will start in 3... 2... 1...\n")
    time.sleep(3)
    
    print('\a')  # Beep
    print("🔴 RECORDING NOW! Speak clearly for 5 seconds...\n")
    
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    
    print("   [OK] Recording captured!\n")
    
    # CORRECT: [1, time]
    waveform = torch.from_numpy(audio.flatten())  # Flatten to 1D [time]
    waveform = waveform.unsqueeze(0)              # Add batch → [1, time]
    return waveform

def enroll_voice(owner_name=None):
    """
    Enroll voice for a specific owner
    
    Args:
        owner_name: Name of owner to enroll. If None, prompts user
        
    Returns:
        Tuple of (owner_name, template_path, success)
    """
    
    if owner_name is None:
        # Get owner name with confirmation
        while True:
            owner_name = input("\n[USER] Enter owner name (e.g., rithika, sid, admin): ").strip().lower()
            if not owner_name:
                print("[ERROR] Owner name cannot be empty!")
                continue
            
            # Confirm owner name
            confirm = input(f"   ✓ Confirm: '{owner_name}'? (y/n): ").strip().lower()
            if confirm == 'y':
                break
            else:
                print("   Please try again\n")

    print(f"\n[RETRY] Loading speaker recognition model for {owner_name}...")
    print("   (This may take 1-2 minutes on first run)\n")

    verification = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        local_strategy=LocalStrategy.COPY
    )
    print("[OK] Model loaded!\n")

    print("=" * 60)
    print(f"[NOTE] ENROLLING VOICE FOR: {owner_name.upper()}")
    print("=" * 60)
    print("\n[WARNING]  IMPORTANT:")
    print("   • Use the EXACT SAME phrase for all 5 samples")
    print("   • Use the SAME tone and pace")
    print("   • Speak clearly and naturally\n")

    if owner_name is None:
        input("Press Enter when ready to start enrollment...\n")

    embeddings = []
    for i in range(1, 6):
        print(f"🔹 Recording sample {i}/5")
        waveform = record(i)
        emb = verification.encode_batch(waveform)
        embeddings.append(emb)
        time.sleep(1)

    print("[RETRY] Creating voice template...")
    voice_template = torch.mean(torch.cat(embeddings, dim=0), dim=0)

    # Save with owner name
    os.makedirs("templates", exist_ok=True)
    template_path = f"templates/{owner_name}_voice_template.pt"
    torch.save(voice_template, template_path)

    print("\n" + "=" * 60)
    print("🎉 ENROLLMENT COMPLETE!")
    print("=" * 60)
    print(f"[OK] Saved: {template_path}")
    print(f"[USER] Owner: {owner_name}")
    print(f"[SECURE] Voice template created with 5 samples\n")

    # List all registered owners
    templates_dir = "templates"
    if os.path.exists(templates_dir):
        template_files = [f for f in os.listdir(templates_dir) 
                          if f.endswith("_voice_template.pt")]
        if template_files:
            print("[INFO] Currently registered owners:")
            for template_file in sorted(template_files):
                owner = template_file.replace("_voice_template.pt", "")
                print(f"   ✓ {owner}")
            print()

    print("[START] Ready to use! Run smart_lock.py to test authentication.\n")
    return owner_name, template_path, True

# Allow running as script
if __name__ == "__main__":
    print("=" * 60)
    print("🔊 MULTI-OWNER VOICE ENROLLMENT SYSTEM")
    print("=" * 60)
    
    try:
        enroll_voice()
    except Exception as e:
        print(f"[ERROR] Enrollment failed: {e}")
