"""
FaceNet Module - Face Recognition and Verification
Provides face embedding generation, verification, and enrollment capabilities
"""
try:
    from .face_embedder import get_face_embedding
    from .verify_face import verify_face
    from .enroll_faces import enroll_faces
    from .capture_live import capture_face
    __all__ = ['get_face_embedding', 'verify_face', 'enroll_faces', 'capture_face']
except ImportError as e:
    print(f"[WARNING] FaceNet imports not fully available: {e}")
    __all__ = []
