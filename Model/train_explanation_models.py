"""
Train and save lightweight explanation models for all owners
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))
from explanation_model import LightweightExplanationModel

def train_all_owners(db_path, model_dir):
    owners = LightweightExplanationModel.available_owners(db_path)
    for owner in owners:
        if owner and owner != 'unknown':
            model = LightweightExplanationModel(db_path, model_dir, owner)
            try:
                model.fit()
            except Exception as e:
                print(f"[ERROR] Could not train model for {owner}: {e}")

if __name__ == "__main__":
    db_path = os.path.abspath("smart_lock_events.db")
    model_dir = os.path.abspath("Model")
    os.makedirs(model_dir, exist_ok=True)
    train_all_owners(db_path, model_dir)
    print("[DONE] All owner models trained.")
