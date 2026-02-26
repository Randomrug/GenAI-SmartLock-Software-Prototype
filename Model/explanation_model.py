"""
Novelty-Driven Lightweight Explanation Model
- Trains a per-owner, context-aware explanation model using smart_lock_events
- Uses a hybrid of decision trees and attention-inspired feature weighting for novelty
- Designed for patentable, adaptive, explainable access control
"""
import os
import sqlite3
import pickle
import numpy as np
from collections import Counter

class LightweightExplanationModel:
    """Adaptive, context-aware explanation model for access control"""
    def __init__(self, db_path, model_dir, owner):
        self.db_path = db_path
        self.model_dir = model_dir
        self.owner = owner.lower()
        self.model_file = os.path.join(model_dir, f"explanation_model_{self.owner}.pkl")
        self.model = None
        self.feature_weights = None

    def extract_features_and_labels(self):
        """Extracts features and explanations from smart_lock_events for the owner"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT day_of_week, action, pin_valid, face_score, voice_score, behavior_score, final_score, genai_decision, genai_risk_level, genai_explanation
            FROM access_events WHERE owner = ?
        """, (self.owner,))
        rows = cursor.fetchall()
        conn.close()
        X, y = [], []
        for row in rows:
            day, action, pin, face, voice, beh, final, decision, risk, explanation = row
            # Encode categorical features
            day_idx = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day) if day else 0
            action_idx = 1 if action == "IN" else 0
            pin = int(pin)
            # Feature vector: [day, action, pin, face, voice, beh, final, decision, risk]
            decision_idx = {"ALLOW":0, "DENY":1, "LOCKOUT":2}.get(decision, 0)
            risk_idx = {"LOW":0, "MEDIUM":1, "HIGH":2}.get(risk, 1)
            X.append([day_idx, action_idx, pin, face, voice, beh, final, decision_idx, risk_idx])
            y.append(explanation)
        return np.array(X), np.array(y)

    def fit(self):
        """Train the model using a hybrid attention-tree approach"""
        X, y = self.extract_features_and_labels()
        if len(X) == 0:
            raise ValueError("No data for owner: " + self.owner)
        # Step 1: Feature weighting (attention-inspired)
        self.feature_weights = self._compute_feature_weights(X, y)
        X_weighted = X * self.feature_weights
        # Step 2: Train a simple decision tree (novelty: use weighted features)
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3)
        self.model.fit(X_weighted, y)
        # Save model
        with open(self.model_file, "wb") as f:
            pickle.dump({"model": self.model, "feature_weights": self.feature_weights}, f)
        print(f"[OK] Trained and saved explanation model for {self.owner} at {self.model_file}")

    def _compute_feature_weights(self, X, y):
        """Novelty: Compute feature weights based on mutual information with explanations (attention-like)"""
        from sklearn.feature_selection import mutual_info_classif
        weights = mutual_info_classif(X, y, discrete_features=[0,1,2,7,8])
        # Normalize and boost most informative features
        weights = np.clip(weights / (weights.max() + 1e-6), 0.2, 1.0)
        print(f"[INFO] Feature weights: {weights}")
        return weights

    def predict(self, context):
        """Predict explanation for a new access attempt context (dict of features), applying explicit owner rules first."""
        owner = context.get("owner", "unknown").lower()
        day = context.get("day_of_week", "Monday")
        time_str = context.get("time", "00:00")
        # Support HH:MM or HH:MM:SS
        time_parts = list(map(int, time_str.split(":")))
        hour, minute = time_parts[0], time_parts[1]
        t_minutes = hour * 60 + minute
        # Debug: print parsed time
        print(f"[DEBUG] Parsed time: {time_str} -> {t_minutes} minutes (owner={owner}, day={day})")
        pin_valid = bool(context.get("pin_valid", 0))
        face_score = float(context.get("face_score", 0))
        voice_score = float(context.get("voice_score", 0))
        final_score = float(context.get("final_score", 0))
        owner_mismatch = context.get("owner_mismatch", False)

        # --- Explicit Owner Rules ---
        decision = None
        risk = None
        explanation = ""

        if owner == "rithika":
            if day == "Monday":
                if 420 <= t_minutes <= 540:  # 07:00–09:00
                    risk = "LOW"
                    explanation = "Rithika Monday morning rush (early college departure)."
                elif 480 <= t_minutes <= 1170:  # 08:00–19:30
                    risk = "LOW"
                    explanation = "Rithika Monday normal entry window."
                elif 780 <= t_minutes <= 840:  # 13:00–14:00
                    risk = "LOW"
                    explanation = "Rithika Monday lunch break re-entry."
                elif 1260 <= t_minutes <= 1290:  # 21:00–21:30
                    risk = "MEDIUM"
                    explanation = "Rithika late Monday entry (after 21:00)."
                elif t_minutes > 1350:  # after 22:30
                    if face_score > 0.85 and voice_score > 0.85:
                        decision = "ALLOW"
                        risk = "MEDIUM"
                        explanation = "Rithika very late Monday entry (after 22:30). Strong biometrics, so access allowed with MEDIUM risk. Mild alert sent."
                        context['trigger_alert'] = True
                        return {"decision": decision, "risk_level": risk, "explanation": explanation, "trigger_alert": True}
                    else:
                        risk = "HIGH"
                        explanation = "Rithika very late Monday entry (after 22:30)."
            elif day in ["Tuesday", "Wednesday", "Thursday", "Friday"]:
                if 480 <= t_minutes <= 1170:  # 08:00–19:30
                    risk = "LOW"
                    explanation = f"Rithika {day} normal entry window."
                elif 780 <= t_minutes <= 840:  # 13:00–14:00
                    risk = "LOW"
                    explanation = f"Rithika {day} lunch break re-entry."
                elif 1260 <= t_minutes <= 1290:  # 21:00–21:30
                    risk = "MEDIUM"
                    explanation = f"Rithika late {day} entry (after 21:00)."
                elif t_minutes > 1350:  # after 22:30
                    if face_score > 0.85 and voice_score > 0.85:
                        decision = "ALLOW"
                        risk = "MEDIUM"
                        explanation = f"Rithika very late {day} entry (after 22:30). Strong biometrics, so access allowed with MEDIUM risk. Mild alert sent."
                        context['trigger_alert'] = True
                        return {"decision": decision, "risk_level": risk, "explanation": explanation, "trigger_alert": True}
                    else:
                        risk = "HIGH"
                        explanation = f"Rithika very late {day} entry (after 22:30)."
            elif day == "Saturday":
                # Accept up to and including 23:30 (1410 minutes), exclusive upper bound for safety
                if 1200 <= t_minutes < 1411:  # 20:00–23:30 inclusive
                    decision = "ALLOW"
                    risk = "MEDIUM"
                    explanation = (
                        f"Access at {context.get('time','')} on Saturday is explicitly permitted for Rithika. This late entry is typical for her on Saturdays (family dinner or movie night), and the pattern is normal."
                    )
                    print(f"[DEBUG] Rithika Saturday late-night rule triggered: {explanation}")
                    return {
                        "decision": decision,
                        "risk_level": risk,
                        "explanation": explanation
                    }
                elif t_minutes < 1200:  # before 20:00
                    risk = "LOW"
                    explanation = "Rithika Saturday normal/flexible schedule."
                else:
                    risk = "HIGH"
                    explanation = "Rithika Saturday entry after 23:30."
            elif day == "Sunday":
                if t_minutes > 1320:  # after 22:00
                    risk = "MEDIUM"
                    explanation = "Rithika Sunday late entry (after 22:00)."
                else:
                    risk = "LOW"
                    explanation = "Rithika Sunday entry."
        elif owner == "sid":
            if day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                if 540 <= t_minutes <= 1020:  # 09:00–17:00
                    risk = "LOW"
                    explanation = "Sid weekday normal school hours."
                elif 780 <= t_minutes <= 840:  # 13:00–14:00
                    risk = "HIGH"
                    explanation = "Sid unexpected lunch re-entry (not expected)."
                elif t_minutes < 540 or t_minutes > 1020:
                    risk = "MEDIUM" if t_minutes > 1260 else "LOW"
                    explanation = "Sid weekday entry outside normal hours."
                if t_minutes > 1260:  # after 21:00
                    risk = "MEDIUM"
                    explanation = "Sid weekday late entry (after 21:00)."
            elif day == "Saturday":
                if t_minutes > 1320:  # after 22:00
                    risk = "MEDIUM"
                    explanation = "Sid Saturday late entry (after 22:00)."
                else:
                    risk = "LOW"
                    explanation = "Sid Saturday flexible schedule."
            elif day == "Sunday":
                if 960 <= t_minutes <= 1080:  # 16:00–18:00
                    risk = "LOW"
                    explanation = "Sid Sunday badminton practice window."
                elif t_minutes > 1290:  # after 21:30
                    risk = "MEDIUM"
                    explanation = "Sid Sunday late entry (after 21:30)."
                else:
                    risk = "LOW"
                    explanation = "Sid Sunday entry."

        # --- Biometric and Rule Fusion ---
        # Hard rules override unless biometrics are extremely strong
        strong_biometrics = (face_score > 0.95 and voice_score > 0.95)
        biometric_mismatch = abs(face_score - voice_score) > 0.2
        if biometric_mismatch:
            final_score *= 0.6
            explanation += " Biometric mismatch detected. Final score reduced."

        if not pin_valid:
            decision = "DENY"
            risk = "HIGH"
            explanation += " PIN invalid. Access denied."
        elif risk == "HIGH" and not strong_biometrics:
            decision = "DENY"
            explanation += " Rule violation and weak biometrics."
        elif risk == "HIGH" and strong_biometrics:
            decision = "ALLOW"
            risk = "MEDIUM"
            explanation += " Rule violation but strong biometrics. Allowing with MEDIUM risk."
        elif final_score < 0.85:
            decision = "DENY"
            risk = risk or "HIGH"
            explanation += " Final score below threshold."
        else:
            decision = "ALLOW"
            risk = risk or "LOW"
            explanation += " Access granted."

        # If no explicit rule matched, fallback to learned model
        if decision is None or risk is None:
            if self.model is None or self.feature_weights is None:
                with open(self.model_file, "rb") as f:
                    obj = pickle.load(f)
                    self.model = obj["model"]
                    self.feature_weights = obj["feature_weights"]
            day_idx = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(context.get("day_of_week","Monday"))
            action_idx = 1 if context.get("action") == "IN" else 0
            pin = int(context.get("pin_valid",0))
            face = float(context.get("face_score",0))
            voice = float(context.get("voice_score",0))
            beh = float(context.get("behavior_score",0))
            final = float(context.get("final_score",0))
            decision_idx = {"ALLOW":0, "DENY":1, "LOCKOUT":2}.get(context.get("genai_decision","ALLOW"),0)
            risk_idx = {"LOW":0, "MEDIUM":1, "HIGH":2}.get(context.get("genai_risk_level","MEDIUM"),1)
            X = np.array([[day_idx, action_idx, pin, face, voice, beh, final, decision_idx, risk_idx]])
            X_weighted = X * self.feature_weights
            pred = self.model.predict(X_weighted)
            explanation += f" [Learned model fallback: {pred[0]}]"
            return pred[0]

        # Compose explanation
        full_explanation = f"Owner: {owner.title()}. {explanation} (Day: {day}, Time: {time_str}, Risk: {risk}, Face: {face_score:.2f}, Voice: {voice_score:.2f}, Final: {final_score:.2f})"
        return {
            "decision": decision,
            "risk_level": risk,
            "explanation": full_explanation
        }

    @staticmethod
    def available_owners(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT owner FROM access_events")
        owners = [row[0] for row in cursor.fetchall()]
        conn.close()
        return owners
