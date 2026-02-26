"""
VoiceRecognition Module - Voice Authentication
Provides voice enrollment and verification capabilities using SpeechBrain
"""
try:
    from .smart_lock import verify_voice, init_voice_system
    from .enroll_my_voice import enroll_voice
    __all__ = ['verify_voice', 'init_voice_system', 'enroll_voice']
except ImportError as e:
    print(f"[WARNING] VoiceRecognition imports not fully available: {e}")
    __all__ = []
