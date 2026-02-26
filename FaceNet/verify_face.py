import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_face(live_embedding, face_db, threshold):
    scores = {
        owner: cosine_similarity(live_embedding, emb)
        for owner, emb in face_db.items()
    }

    best_owner = max(scores, key=scores.get)
    best_score = scores[best_owner]

    return best_owner, best_score, best_score >= threshold
