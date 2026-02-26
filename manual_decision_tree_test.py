from Model.explanation_model import LightweightExplanationModel
import os

# Test Rithika, Saturday, 23:04
model = LightweightExplanationModel(
    db_path=os.path.abspath("smart_lock_events.db"),
    model_dir=os.path.abspath("Model"),
    owner="rithika"
)

context = {
    "owner": "rithika",
    "day_of_week": "Saturday",
    "time": "23:04",
    "action": "IN",
    "pin_valid": 1,
    "face_score": 0.95,
    "voice_score": 0.75,
    "behavior_score": 0.6,
    "final_score": 0.9,
    "genai_decision": None,
    "genai_risk_level": None
}

result = model.predict(context)
print("Manual Decision Tree Test Result:")
print(result)
