"""
Anomaly Detection Module
Detects unusual patterns and suspicious activities using ML algorithms
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Any

class AnomalyDetector:
    """
    Anomaly detection system for smart lock access patterns
    Uses multiple algorithms for robust anomaly detection
    """
    
    def __init__(self, model_dir: str = 'models/anomaly_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.initialize_models()
        
        # Feature configuration
        self.features = {
            'temporal': ['hour', 'day_of_week', 'is_weekend'],
            'behavioral': ['face_score', 'voice_score', 'final_score', 'pin_valid'],
            'contextual': ['failed_attempts', 'risk_level_numeric', 'time_since_last_access']
        }
        
        # Load or create models
        self.load_models()
    
    def initialize_models(self):
        """Initialize ML models for anomaly detection"""
        # Isolation Forest - good for high-dimensional data
        self.models['isolation_forest'] = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Expected proportion of outliers
            random_state=42,
            n_jobs=-1
        )
        
        # One-Class SVM - good for novelty detection
        self.models['one_class_svm'] = OneClassSVM(
            nu=0.1,  # Upper bound on fraction of training errors
            kernel='rbf',
            gamma='auto'
        )
        
        # Local Outlier Factor - density-based
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True,
            n_jobs=-1
        )
        
        # Scaler for feature normalization
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def extract_features(self, events: List[Dict]) -> pd.DataFrame:
        """
        Extract features from access events for anomaly detection
        
        Args:
            events: List of access event dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        if not events:
            return pd.DataFrame()
        
        features_list = []
        
        for i, event in enumerate(events):
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(event['record_created_at'].replace('Z', '+00:00'))
                
                # Temporal features
                hour = timestamp.hour
                day_of_week = timestamp.weekday()  # Monday=0, Sunday=6
                is_weekend = 1 if day_of_week >= 5 else 0
                
                # Behavioral features
                face_score = event.get('face_score', 0.5)
                voice_score = event.get('voice_score', 0.5)
                final_score = event.get('final_score', 0.5)
                pin_valid = 1 if event.get('pin_valid') else 0
                
                # Contextual features
                failed_attempts = event.get('failed_attempt_count', 0)
                
                # Convert risk level to numeric
                risk_level = event.get('genai_risk_level', 'MEDIUM')
                risk_numeric = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}.get(risk_level, 1)
                
                # Time since last access (if available)
                time_since_last = 0
                if i > 0:
                    prev_time = datetime.fromisoformat(events[i-1]['record_created_at'].replace('Z', '+00:00'))
                    time_since_last = (timestamp - prev_time).total_seconds() / 3600  # hours
                
                # Create feature vector
                features = {
                    'hour': hour,
                    'day_of_week': day_of_week,
                    'is_weekend': is_weekend,
                    'face_score': face_score,
                    'voice_score': voice_score,
                    'final_score': final_score,
                    'pin_valid': pin_valid,
                    'failed_attempts': failed_attempts,
                    'risk_level_numeric': risk_numeric,
                    'time_since_last_access': time_since_last,
                    
                    # Derived features
                    'hour_sin': np.sin(2 * np.pi * hour / 24),
                    'hour_cos': np.cos(2 * np.pi * hour / 24),
                    'day_sin': np.sin(2 * np.pi * day_of_week / 7),
                    'day_cos': np.cos(2 * np.pi * day_of_week / 7),
                    'score_variance': abs(face_score - voice_score),
                    'total_score': (face_score + voice_score) / 2
                }
                
                features_list.append(features)
                
            except Exception as e:
                print(f"Error extracting features from event {i}: {e}")
                continue
        
        return pd.DataFrame(features_list)
    
    def train_models(self, training_events: List[Dict], save_models: bool = True):
        """
        Train anomaly detection models on historical data
        
        Args:
            training_events: Historical access events for training
            save_models: Whether to save trained models to disk
        """
        print("Training anomaly detection models...")
        
        # Extract features
        X = self.extract_features(training_events)
        
        if X.empty or len(X) < 10:
            print("[WARNING]  Insufficient data for training")
            return
        
        print(f"  Training with {len(X)} samples, {X.shape[1]} features")
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                print(f"  Training {model_name}...")
                
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(X)
                
                # Train model
                if model_name == 'lof':
                    model.fit(X_scaled)
                else:
                    model.fit(X_scaled)
                
                # Calculate threshold (for scoring)
                scores = self._calculate_anomaly_scores(model, X_scaled, model_name)
                self.thresholds[model_name] = np.percentile(scores, 95)  # 95th percentile as threshold
                
                print(f"    ✓ {model_name} trained (threshold: {self.thresholds[model_name]:.3f})")
                
            except Exception as e:
                print(f"    ✗ Error training {model_name}: {e}")
        
        # Save models if requested
        if save_models:
            self.save_models()
        
        print("[OK] Anomaly detection models trained successfully")
    
    def detect_anomalies(self, current_event: Dict, historical_context: List[Dict] = None) -> Dict:
        """
        Detect anomalies in current access attempt
        
        Args:
            current_event: Current access attempt
            historical_context: Recent historical events for context
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Prepare data for detection
        if historical_context:
            events_for_detection = historical_context + [current_event]
        else:
            events_for_detection = [current_event]
        
        # Extract features
        X = self.extract_features(events_for_detection)
        
        if X.empty:
            return self._create_default_result()
        
        # Get features for current event (last in sequence)
        current_features = X.iloc[-1:].values
        
        results = {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'confidence': 0.0,
            'model_predictions': {},
            'anomaly_reasons': [],
            'feature_contributions': {}
        }
        
        # Get predictions from each model
        model_scores = []
        
        for model_name, model in self.models.items():
            try:
                # Scale features
                X_scaled = self.scalers[model_name].transform(current_features)
                
                # Get prediction
                if model_name == 'isolation_forest':
                    score = model.score_samples(X_scaled)[0]
                    is_anomaly = score < self.thresholds.get(model_name, -0.5)
                    
                elif model_name == 'one_class_svm':
                    prediction = model.predict(X_scaled)[0]
                    is_anomaly = prediction == -1
                    score = -model.decision_function(X_scaled)[0]  # Negative distance to decision boundary
                    
                elif model_name == 'lof':
                    score = -model.score_samples(X_scaled)[0]
                    is_anomaly = score < self.thresholds.get(model_name, -1.5)
                
                # Store model prediction
                results['model_predictions'][model_name] = {
                    'score': float(score),
                    'is_anomaly': bool(is_anomaly),
                    'threshold': float(self.thresholds.get(model_name, 0))
                }
                
                model_scores.append(score)
                
                # If any model detects anomaly, add to reasons
                if is_anomaly:
                    results['anomaly_reasons'].append(f"{model_name} detected unusual pattern")
                    
            except Exception as e:
                print(f"Error in {model_name} prediction: {e}")
                continue
        
        # Calculate ensemble results
        if model_scores:
            # Average score across models
            results['anomaly_score'] = float(np.mean(model_scores))
            
            # Calculate confidence based on model agreement
            anomalies_detected = sum(1 for pred in results['model_predictions'].values() 
                                   if pred.get('is_anomaly', False))
            total_models = len(results['model_predictions'])
            
            if total_models > 0:
                results['confidence'] = anomalies_detected / total_models
                results['is_anomaly'] = results['confidence'] >= 0.5  # Majority voting
            
            # Analyze feature contributions
            results['feature_contributions'] = self._analyze_feature_contributions(
                current_features[0], X.columns
            )
        
        # Add contextual analysis
        results.update(self._contextual_analysis(current_event, historical_context))
        
        return results
    
    def _calculate_anomaly_scores(self, model, X_scaled: np.ndarray, model_name: str) -> np.ndarray:
        """Calculate anomaly scores for given model"""
        if model_name == 'isolation_forest':
            return model.score_samples(X_scaled)
        elif model_name == 'one_class_svm':
            return -model.decision_function(X_scaled)
        elif model_name == 'lof':
            return -model.score_samples(X_scaled)
        else:
            return np.zeros(len(X_scaled))
    
    def _analyze_feature_contributions(self, features: np.ndarray, feature_names: List[str]) -> Dict:
        """Analyze which features contributed most to anomaly detection"""
        contributions = {}
        
        # Simple heuristic: features far from typical range contribute more
        typical_ranges = {
            'hour': (7, 22),  # Normal hours: 7 AM - 10 PM
            'face_score': (0.7, 1.0),
            'voice_score': (0.7, 1.0),
            'final_score': (0.7, 1.0),
            'failed_attempts': (0, 2),
            'risk_level_numeric': (0, 1)  # LOW or MEDIUM
        }
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in typical_ranges:
                min_val, max_val = typical_ranges[feature_name]
                feature_val = features[i]
                
                # Calculate how far outside typical range
                if feature_val < min_val:
                    deviation = (min_val - feature_val) / min_val
                elif feature_val > max_val:
                    deviation = (feature_val - max_val) / max_val
                else:
                    deviation = 0
                
                if deviation > 0.1:  # Significant deviation
                    contributions[feature_name] = {
                        'value': float(feature_val),
                        'typical_range': (float(min_val), float(max_val)),
                        'deviation': float(deviation)
                    }
        
        # Sort by deviation (highest first)
        sorted_contributions = dict(sorted(
            contributions.items(),
            key=lambda x: x[1]['deviation'],
            reverse=True
        ))
        
        return sorted_contributions
    
    def _contextual_analysis(self, current_event: Dict, historical_context: List[Dict] = None) -> Dict:
        """Perform contextual anomaly analysis"""
        contextual_results = {
            'contextual_anomalies': [],
            'temporal_pattern': 'normal',
            'behavioral_consistency': 'high'
        }
        
        if not historical_context:
            return contextual_results
        
        try:
            # Analyze temporal pattern
            current_time = datetime.fromisoformat(
                current_event['record_created_at'].replace('Z', '+00:00')
            )
            current_hour = current_time.hour
            
            # Check if unusual hour
            if current_hour < 5 or current_hour > 23:  # Very early morning or late night
                contextual_results['contextual_anomalies'].append('Unusual access time')
                contextual_results['temporal_pattern'] = 'unusual'
            
            # Analyze behavioral consistency
            recent_successful = [
                e for e in historical_context[-10:]  # Last 10 events
                if e.get('genai_decision') == 'ALLOW'
            ]
            
            if recent_successful:
                avg_face = np.mean([e.get('face_score', 0) for e in recent_successful])
                avg_voice = np.mean([e.get('voice_score', 0) for e in recent_successful])
                
                current_face = current_event.get('face_score', 0)
                current_voice = current_event.get('voice_score', 0)
                
                # Check for significant deviation
                face_deviation = abs(current_face - avg_face) / avg_face if avg_face > 0 else 1
                voice_deviation = abs(current_voice - avg_voice) / avg_voice if avg_voice > 0 else 1
                
                if face_deviation > 0.3 or voice_deviation > 0.3:
                    contextual_results['contextual_anomalies'].append('Inconsistent biometric scores')
                    contextual_results['behavioral_consistency'] = 'low'
            
            # Check for rapid succession attempts
            if len(historical_context) >= 2:
                last_time = datetime.fromisoformat(
                    historical_context[-1]['record_created_at'].replace('Z', '+00:00')
                )
                time_diff = (current_time - last_time).total_seconds()
                
                if time_diff < 30:  # Less than 30 seconds between attempts
                    contextual_results['contextual_anomalies'].append('Rapid succession attempts')
                    contextual_results['temporal_pattern'] = 'rapid'
            
        except Exception as e:
            print(f"Error in contextual analysis: {e}")
        
        return contextual_results
    
    def _create_default_result(self) -> Dict:
        """Create default result when detection fails"""
        return {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'confidence': 0.0,
            'model_predictions': {},
            'anomaly_reasons': ['Insufficient data for analysis'],
            'feature_contributions': {},
            'contextual_anomalies': [],
            'temporal_pattern': 'unknown',
            'behavioral_consistency': 'unknown'
        }
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f'{model_name}.joblib')
                joblib.dump(model, model_path)
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = os.path.join(self.model_dir, f'{scaler_name}_scaler.joblib')
                joblib.dump(scaler, scaler_path)
            
            # Save thresholds
            thresholds_path = os.path.join(self.model_dir, 'thresholds.json')
            with open(thresholds_path, 'w') as f:
                json.dump(self.thresholds, f, indent=2)
            
            print(f"[BACKUP] Models saved to {self.model_dir}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            for model_name in self.models.keys():
                model_path = os.path.join(self.model_dir, f'{model_name}.joblib')
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    print(f"  ✓ Loaded {model_name}")
            
            # Load scalers
            for scaler_name in self.scalers.keys():
                scaler_path = os.path.join(self.model_dir, f'{scaler_name}_scaler.joblib')
                if os.path.exists(scaler_path):
                    self.scalers[scaler_name] = joblib.load(scaler_path)
            
            # Load thresholds
            thresholds_path = os.path.join(self.model_dir, 'thresholds.json')
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    self.thresholds = json.load(f)
            
            print(f"[FOLDER] Loaded models from {self.model_dir}")
            
        except Exception as e:
            print(f"[WARNING]  Could not load models, using fresh models: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about trained models"""
        info = {
            'models_trained': list(self.models.keys()),
            'thresholds': self.thresholds,
            'features_used': list(self.features.keys()),
            'model_status': {}
        }
        
        for model_name, model in self.models.items():
            info['model_status'][model_name] = {
                'is_fitted': hasattr(model, 'estimators_') or hasattr(model, 'support_vectors_'),
                'threshold': self.thresholds.get(model_name, 'Not set'),
                'scaler_available': model_name in self.scalers and hasattr(self.scalers[model_name], 'mean_')
            }
        
        return info

# Factory function for easy integration
def create_anomaly_detector(model_dir: str = 'models/anomaly_models') -> AnomalyDetector:
    """Factory function to create and initialize anomaly detector"""
    detector = AnomalyDetector(model_dir)
    return detector

if __name__ == "__main__":
    # Test the anomaly detector
    print("[TEST] Testing Anomaly Detector")
    
    # Create sample events for testing
    sample_events = []
    base_time = datetime.now()
    
    for i in range(100):
        event_time = base_time - timedelta(hours=i)
        
        sample_events.append({
            'record_created_at': event_time.isoformat(),
            'face_score': 0.8 + np.random.normal(0, 0.1),
            'voice_score': 0.75 + np.random.normal(0, 0.1),
            'final_score': 0.78 + np.random.normal(0, 0.08),
            'pin_valid': True,
            'failed_attempt_count': 0,
            'genai_risk_level': 'LOW',
            'genai_decision': 'ALLOW'
        })
    
    # Add some anomalies
    for i in range(10):
        event_time = base_time - timedelta(hours=i, minutes=30)
        
        sample_events.append({
            'record_created_at': event_time.isoformat(),
            'face_score': 0.3 + np.random.normal(0, 0.2),
            'voice_score': 0.4 + np.random.normal(0, 0.2),
            'final_score': 0.35 + np.random.normal(0, 0.15),
            'pin_valid': False,
            'failed_attempt_count': 2,
            'genai_risk_level': 'HIGH',
            'genai_decision': 'DENY'
        })
    
    # Create and test detector
    detector = AnomalyDetector()
    
    # Train models
    detector.train_models(sample_events[:80], save_models=False)
    
    # Test detection
    test_event = {
        'record_created_at': datetime.now().isoformat(),
        'face_score': 0.2,  # Very low - should be anomalous
        'voice_score': 0.9,
        'final_score': 0.55,
        'pin_valid': True,
        'failed_attempt_count': 0,
        'genai_risk_level': 'MEDIUM',
        'genai_decision': 'ALLOW'
    }
    
    result = detector.detect_anomalies(test_event, sample_events[:20])
    
    print("\n[SEARCH] Anomaly Detection Result:")
    print(json.dumps(result, indent=2))
    
    print(f"\n[CHART] Model Info:")
    print(json.dumps(detector.get_model_info(), indent=2))