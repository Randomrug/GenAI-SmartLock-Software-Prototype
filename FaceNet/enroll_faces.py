import os
import pickle
import numpy as np
from face_embedder import get_face_embedding

ENROLL_DIR = "enroll"
OUT_FILE = "embeddings/face_db.pkl"

def enroll_faces(enroll_dir=ENROLL_DIR, out_file=OUT_FILE):
    """Enroll faces from directory"""
    face_db = {}

    for owner in os.listdir(enroll_dir):
        owner_path = os.path.join(enroll_dir, owner)
        if not os.path.isdir(owner_path):
            continue

        embeddings = []

        for img_name in os.listdir(owner_path):
            img_path = os.path.join(owner_path, img_name)
            emb = get_face_embedding(img_path)

            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            face_db[owner] = np.mean(embeddings, axis=0)
            print(f"[OK] Enrolled {owner}")

    os.makedirs("embeddings", exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(face_db, f)

    print("\n[OK] Face enrollment complete")
    return face_db

# Allow running as script
if __name__ == "__main__":
    enroll_faces()
