import pickle
from capture_live import capture_face_image
from face_embedder import get_face_embedding
from verify_face import verify_face
from config import FACE_SIMILARITY_THRESHOLD

DB_PATH = "embeddings/face_db.pkl"
LIVE_IMAGE = "live.jpg"

print("[LOCK] Face Lock System (Live Camera)")

ok = capture_face_image(LIVE_IMAGE)
if not ok:
    exit()

live_emb = get_face_embedding(LIVE_IMAGE)
if live_emb is None:
    print("[ERROR] No face detected")
    exit()

with open(DB_PATH, "rb") as f:
    face_db = pickle.load(f)

owner, score, allowed = verify_face(
    live_emb,
    face_db,
    FACE_SIMILARITY_THRESHOLD
)

print("\nRESULT")
print("Closest Owner :", owner)
print("Similarity    :", round(score, 3))

if allowed:
    print("[OK] ACCESS GRANTED")
else:
    print("[ERROR] ACCESS DENIED")
