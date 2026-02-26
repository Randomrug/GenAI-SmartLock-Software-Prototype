"""
Score Fusion Module
Intelligently combines multiple authentication scores using advanced fusion techniques
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

class ScoreFusion:
    """
    Advanced score fusion system for multi-modal authentication
    Uses machine learning to optimally combine scores
    """
    
    def __init__(self, model_dir: str = 'models/fusion_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Fusion models
        self.models = {}
        self.scalers = {}
        self.fusion_methods = {}
        self.weights = {}
        
        # Configuration
        self.config = {
            'methods': {
                'weighted_average': {
                    'enabled': True,
                    'default_weights': {'pin': 0.3, 'face': 0.35, 'voice': 0.35},
                    'adaptive': True
                },
                'machine_learning': {
                    'enabled': True,
                    'models': ['random_forest', 'gradient_boosting', 'svm'],
                    'default_model': 'random_forest',
                    'retrain_interval': 100  # Retrain after N new samples
                },
                'rule_based': {
                    'enabled': True,
                    'rules': ['minimum_threshold', 'consistency_check', 'context_adjustment']
                }
            },
            'thresholds': {
                'pin': 0.5,
                'face': 0.6,
                'voice': 0.6,
                'final': 0.7,
                'high_confidence': 0.85
            },
            'calibration': {
                'enabled': True,
                'method': 'isotonic',  # 'isotonic', 'sigmoid', 'none'
                'calibration_samples': 100
            }
        }
        
        # Initialize models
        self.initialize_models()
        self.load_models()
    
    def initialize_models(self):
        """Initialize ML models for score fusion"""
        # Random Forest - robust, handles non-linear relationships
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Gradient Boosting - good for complex patterns
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # SVM - good for high-dimensional spaces
        self.models['svm'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        # Scaler for feature normalization
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def prepare_features(self, scores: Dict, context: Dict = None) -> np.ndarray:
        """
        Prepare features for fusion models
        
        Args:
            scores: Dictionary with individual scores
            context: Additional context information
            
        Returns:
            Feature vector for fusion
        """
        # Base scores
        pin_score = 1.0 if scores.get('pin_valid', False) else 0.0
        face_score = scores.get('face_score', 0.0)
        voice_score = scores.get('voice_score', 0.0)
        
        # Derived features
        score_mean = (face_score + voice_score) / 2
        score_variance = abs(face_score - voice_score)
        score_consistency = 1 - score_variance
        
        # Create feature vector
        features = [
            pin_score,
            face_score,
            voice_score,
            score_mean,
            score_variance,
            score_consistency,
            
            # Additional features from context
            scores.get('behavior_score', 0.5),
            scores.get('failed_attempt_count', 0) / 10,  # Normalized
            1 if scores.get('lockout_active', False) else 0
        ]
        
        # Add time-based features if available
        if context and 'access_time' in context:
            try:
                access_time = context['access_time']
                if isinstance(access_time, str):
                    access_time = datetime.fromisoformat(access_time.replace('Z', '+00:00'))
                
                hour = access_time.hour
                
                # Time features
                features.extend([
                    hour / 24,  # Normalized hour
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    1 if 7 <= hour <= 22 else 0,  # Normal hours
                ])
            except:
                features.extend([0.5, 0, 1, 1])  # Default values
        
        # Ensure consistent feature length
        while len(features) < 15:  # Pad to minimum length
            features.append(0.0)
        
        return np.array(features).reshape(1, -1)
    
    def weighted_average_fusion(self, scores: Dict, weights: Dict = None) -> Dict:
        """
        Weighted average fusion method
        
        Args:
            scores: Individual scores
            weights: Custom weights (if None, use configured weights)
            
        Returns:
            Fusion result
        """
        if weights is None:
            weights = self.config['methods']['weighted_average']['default_weights']
        
        # Get individual scores
        pin_valid = scores.get('pin_valid', False)
        face_score = scores.get('face_score', 0.0)
        voice_score = scores.get('voice_score', 0.0)
        behavior_score = scores.get('behavior_score', 0.5)
        
        # Apply thresholds
        face_valid = face_score >= self.config['thresholds']['face']
        voice_valid = voice_score >= self.config['thresholds']['voice']
        
        # Calculate weighted score
        weighted_score = (
            weights['pin'] * (1.0 if pin_valid else 0.0) +
            weights['face'] * face_score +
            weights['voice'] * voice_score
        )
        
        # Apply behavior adjustment
        adjusted_score = weighted_score * behavior_score
        
        # Check individual thresholds
        all_thresholds_passed = (
            pin_valid and
            face_valid and
            voice_valid and
            adjusted_score >= self.config['thresholds']['final']
        )
        
        return {
            'method': 'weighted_average',
            'final_score': float(adjusted_score),
            'weighted_score': float(weighted_score),
            'behavior_adjusted': float(behavior_score),
            'thresholds_passed': {
                'pin': pin_valid,
                'face': face_valid,
                'voice': voice_valid,
                'final': adjusted_score >= self.config['thresholds']['final']
            },
            'all_thresholds_passed': all_thresholds_passed,
            'confidence': min(1.0, adjusted_score * 1.2)  # Slight boost for confidence
        }
    
    def machine_learning_fusion(self, scores: Dict, context: Dict = None, 
                              model_name: str = None) -> Dict:
        """
        Machine learning-based fusion
        
        Args:
            scores: Individual scores
            context: Additional context
            model_name: Specific model to use
            
        Returns:
            Fusion result
        """
        if model_name is None:
            model_name = self.config['methods']['machine_learning']['default_model']
        
        if model_name not in self.models:
            return self.weighted_average_fusion(scores)
        
        # Prepare features
        features = self.prepare_features(scores, context)
        
        # Scale features
        scaler = self.scalers.get(model_name)
        if scaler is not None and hasattr(scaler, 'mean_'):
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
        
        # Get model
        model = self.models[model_name]
        
        try:
            # Check if model is trained
            if not hasattr(model, 'classes_'):
                return self.weighted_average_fusion(scores)
            
            # Predict probability
            proba = model.predict_proba(features_scaled)[0]
            
            # Get class indices (assuming binary classification: 0=deny, 1=allow)
            if len(model.classes_) >= 2:
                deny_idx = np.where(model.classes_ == 0)[0]
                allow_idx = np.where(model.classes_ == 1)[0]
                
                deny_prob = proba[deny_idx[0]] if len(deny_idx) > 0 else 0
                allow_prob = proba[allow_idx[0]] if len(allow_idx) > 0 else 0
                
                final_score = allow_prob
                prediction = 1 if allow_prob > 0.5 else 0
            else:
                # Fallback to regression or binary output
                prediction = model.predict(features_scaled)[0]
                final_score = prediction if isinstance(prediction, (int, float)) else 0.5
            
            # Calculate confidence based on probability margin
            confidence = abs(allow_prob - deny_prob) if 'allow_prob' in locals() and 'deny_prob' in locals() else 0.5
            
            return {
                'method': f'ml_{model_name}',
                'final_score': float(final_score),
                'prediction': int(prediction),
                'allow_probability': float(allow_prob) if 'allow_prob' in locals() else float(final_score),
                'deny_probability': float(deny_prob) if 'deny_prob' in locals() else 1 - float(final_score),
                'confidence': float(confidence),
                'threshold_passed': final_score >= self.config['thresholds']['final'],
                'model_used': model_name,
                'model_trained': True
            }
            
        except Exception as e:
            print(f"ML fusion error ({model_name}): {e}")
            return self.weighted_average_fusion(scores)
    
    def rule_based_fusion(self, scores: Dict, context: Dict = None) -> Dict:
        """
        Rule-based fusion with logical rules
        
        Args:
            scores: Individual scores
            context: Additional context
            
        Returns:
            Fusion result
        """
        # Get scores
        pin_valid = scores.get('pin_valid', False)
        face_score = scores.get('face_score', 0.0)
        voice_score = scores.get('voice_score', 0.0)
        behavior_score = scores.get('behavior_score', 0.5)
        failed_attempts = scores.get('failed_attempt_count', 0)
        risk_level = scores.get('risk_level', 'MEDIUM')
        
        # Apply rules
        rules_passed = []
        rules_failed = []
        
        # Rule 1: PIN must be valid for high-security access
        if pin_valid:
            rules_passed.append('PIN valid')
        else:
            rules_failed.append('PIN invalid')
        
        # Rule 2: At least one biometric must be strong
        strong_face = face_score >= 0.8
        strong_voice = voice_score >= 0.8
        moderate_face = face_score >= 0.6
        moderate_voice = voice_score >= 0.6
        
        if strong_face or strong_voice:
            rules_passed.append('Strong biometric present')
        elif moderate_face and moderate_voice:
            rules_passed.append('Moderate biometrics present')
        else:
            rules_failed.append('Insufficient biometrics')
        
        # Rule 3: Consistency check
        score_difference = abs(face_score - voice_score)
        if score_difference < 0.3:
            rules_passed.append('Biometric consistency')
        else:
            rules_failed.append('Biometric inconsistency')
        
        # Rule 4: Behavior pattern
        if behavior_score >= 0.7:
            rules_passed.append('Normal behavior pattern')
        elif behavior_score >= 0.4:
            rules_passed.append('Moderate behavior pattern')
        else:
            rules_failed.append('Atypical behavior')
        
        # Rule 5: Risk level consideration
        risk_factor = {'LOW': 1.0, 'MEDIUM': 0.8, 'HIGH': 0.5}.get(risk_level, 0.8)
        
        # Rule 6: Failure streak penalty
        failure_penalty = 1.0 / (1.0 + failed_attempts * 0.5)
        
        # Calculate rule-based score
        base_score = (face_score + voice_score) / 2
        rule_score = base_score * risk_factor * failure_penalty * behavior_score
        
        # Adjust based on rules passed
        rule_adjustment = len(rules_passed) / (len(rules_passed) + len(rules_failed))
        final_score = rule_score * rule_adjustment
        
        # Determine if access should be allowed
        critical_rules_passed = (
            pin_valid and
            (moderate_face or moderate_voice) and
            behavior_score >= 0.4
        )
        
        return {
            'method': 'rule_based',
            'final_score': float(final_score),
            'base_score': float(base_score),
            'rule_adjustment': float(rule_adjustment),
            'risk_factor': float(risk_factor),
            'failure_penalty': float(failure_penalty),
            'rules_passed': rules_passed,
            'rules_failed': rules_failed,
            'critical_rules_passed': critical_rules_passed,
            'access_recommended': critical_rules_passed and final_score >= 0.6,
            'confidence': min(1.0, len(rules_passed) / max(len(rules_passed) + len(rules_failed), 1))
        }
    
    def ensemble_fusion(self, scores: Dict, context: Dict = None) -> Dict:
        """
        Ensemble fusion combining multiple methods
        
        Args:
            scores: Individual scores
            context: Additional context
            
        Returns:
            Ensemble fusion result
        """
        results = {}
        
        # Get results from all enabled methods
        if self.config['methods']['weighted_average']['enabled']:
            results['weighted_average'] = self.weighted_average_fusion(scores)
        
        if self.config['methods']['machine_learning']['enabled']:
            ml_model = self.config['methods']['machine_learning']['default_model']
            results['machine_learning'] = self.machine_learning_fusion(scores, context, ml_model)
        
        if self.config['methods']['rule_based']['enabled']:
            results['rule_based'] = self.rule_based_fusion(scores, context)
        
        if not results:
            return self.weighted_average_fusion(scores)
        
        # Extract final scores from each method
        final_scores = []
        confidences = []
        
        for method, result in results.items():
            if 'final_score' in result:
                final_scores.append(result['final_score'])
                confidences.append(result.get('confidence', 0.5))
        
        # Calculate ensemble score (weighted by confidence)
        if final_scores:
            # Use confidence as weights
            weights = np.array(confidences) / np.sum(confidences)
            ensemble_score = np.average(final_scores, weights=weights)
            
            # Calculate ensemble confidence
            ensemble_confidence = np.mean(confidences)
            
            # Determine if access should be allowed
            # Use majority voting on threshold decisions
            threshold_decisions = []
            for result in results.values():
                if 'threshold_passed' in result:
                    threshold_decisions.append(result['threshold_passed'])
                elif 'access_recommended' in result:
                    threshold_decisions.append(result['access_recommended'])
                elif 'all_thresholds_passed' in result:
                    threshold_decisions.append(result['all_thresholds_passed'])
            
            access_allowed = sum(threshold_decisions) >= len(threshold_decisions) / 2
            
        else:
            ensemble_score = 0.5
            ensemble_confidence = 0.5
            access_allowed = False
        
        return {
            'method': 'ensemble',
            'final_score': float(ensemble_score),
            'confidence': float(ensemble_confidence),
            'access_recommended': access_allowed,
            'component_results': results,
            'component_scores': [float(s) for s in final_scores],
            'component_confidences': [float(c) for c in confidences]
        }
    
    def adaptive_fusion(self, scores: Dict, context: Dict = None, 
                       historical_performance: Dict = None) -> Dict:
        """
        Adaptive fusion that adjusts based on historical performance
        
        Args:
            scores: Individual scores
            context: Additional context
            historical_performance: Historical performance of different methods
            
        Returns:
            Adaptive fusion result
        """
        # Start with ensemble fusion
        ensemble_result = self.ensemble_fusion(scores, context)
        
        # Adjust based on historical performance if available
        if historical_performance and 'method_performance' in historical_performance:
            perf = historical_performance['method_performance']
            
            # Calculate method weights based on historical accuracy
            method_weights = {}
            total_weight = 0
            
            for method, stats in perf.items():
                if 'accuracy' in stats and stats['accuracy'] > 0:
                    weight = stats['accuracy'] * stats.get('confidence', 0.5)
                    method_weights[method] = weight
                    total_weight += weight
            
            if total_weight > 0:
                # Re-weight component scores
                component_scores = []
                component_weights = []
                
                for method, result in ensemble_result.get('component_results', {}).items():
                    if method in method_weights and 'final_score' in result:
                        component_scores.append(result['final_score'])
                        component_weights.append(method_weights[method])
                
                if component_scores:
                    # Calculate adaptive score
                    adaptive_score = np.average(component_scores, weights=component_weights)
                    
                    # Update result
                    ensemble_result['final_score'] = float(adaptive_score)
                    ensemble_result['method'] = 'adaptive_ensemble'
                    ensemble_result['method_weights'] = method_weights
        
        # Apply calibration if enabled
        if self.config['calibration']['enabled']:
            ensemble_result = self.calibrate_score(ensemble_result, context)
        
        return ensemble_result
    
    def calibrate_score(self, fusion_result: Dict, context: Dict = None) -> Dict:
        """
        Calibrate fusion score to improve probability estimation
        
        Args:
            fusion_result: Raw fusion result
            context: Additional context for calibration
            
        Returns:
            Calibrated result
        """
        raw_score = fusion_result.get('final_score', 0.5)
        confidence = fusion_result.get('confidence', 0.5)
        
        # Simple sigmoid calibration
        calibrated_score = 1 / (1 + np.exp(-10 * (raw_score - 0.5)))
        
        # Adjust based on confidence
        adjusted_score = calibrated_score * confidence + raw_score * (1 - confidence)
        
        # Ensure score stays in [0, 1] range
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        # Update result
        fusion_result['final_score'] = float(adjusted_score)
        fusion_result['raw_score'] = float(raw_score)
        fusion_result['calibrated_score'] = float(calibrated_score)
        fusion_result['calibration_applied'] = True
        
        return fusion_result
    
    def train_ml_models(self, training_data: List[Dict], labels: List[int]):
        """
        Train machine learning models on labeled data
        
        Args:
            training_data: List of feature dictionaries
            labels: List of labels (0=deny, 1=allow)
        """
        if not training_data or not labels:
            print("[WARNING]  No training data provided")
            return
        
        print("Training ML fusion models...")
        
        # Prepare features and labels
        X = []
        y = np.array(labels)
        
        for data in training_data:
            features = self.prepare_features(data['scores'], data.get('context'))
            X.append(features.flatten())
        
        X = np.array(X)
        
        if len(X) < 10:
            print(f"[WARNING]  Insufficient training samples: {len(X)}")
            return
        
        print(f"  Training with {len(X)} samples, {X.shape[1]} features")
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                print(f"  Training {model_name}...")
                
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(X)
                
                # Train with cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
                
                # Final training
                model.fit(X_scaled, y)
                
                print(f"    ✓ {model_name} trained (CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f})")
                
            except Exception as e:
                print(f"    ✗ Error training {model_name}: {e}")
        
        # Save trained models
        self.save_models()
        
        print("[OK] ML fusion models trained successfully")
    
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
            
            # Save configuration
            config_path = os.path.join(self.model_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"[BACKUP] Fusion models saved to {self.model_dir}")
            
        except Exception as e:
            print(f"Error saving fusion models: {e}")
    
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
            
            # Load configuration
            config_path = os.path.join(self.model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            
            print(f"[FOLDER] Loaded fusion models from {self.model_dir}")
            
        except Exception as e:
            print(f"[WARNING]  Could not load fusion models: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about fusion models"""
        info = {
            'methods_enabled': {
                method: config['enabled']
                for method, config in self.config['methods'].items()
            },
            'thresholds': self.config['thresholds'],
            'ml_models_status': {},
            'calibration_enabled': self.config['calibration']['enabled']
        }
        
        for model_name, model in self.models.items():
            info['ml_models_status'][model_name] = {
                'is_fitted': hasattr(model, 'classes_') or hasattr(model, 'n_features_in_'),
                'feature_count': getattr(model, 'n_features_in_', 'Unknown'),
                'scaler_available': model_name in self.scalers and hasattr(self.scalers[model_name], 'mean_')
            }
        
        return info

# Factory function
def create_score_fusion(model_dir: str = 'models/fusion_models') -> ScoreFusion:
    """Factory function to create score fusion system"""
    return ScoreFusion(model_dir)

if __name__ == "__main__":
    # Test the score fusion system
    print("[TEST] Testing Score Fusion")
    
    # Create fusion system
    fusion = ScoreFusion()
    
    # Test scores
    test_scores = {
        'pin_valid': True,
        'face_score': 0.85,
        'voice_score': 0.78,
        'behavior_score': 0.9,
        'failed_attempt_count': 0,
        'risk_level': 'LOW'
    }
    
    test_context = {
        'access_time': datetime.now().isoformat(),
        'location': 'main_entrance',
        'device': 'mobile'
    }
    
    # Test different fusion methods
    print("\n[SEARCH] Testing Fusion Methods:")
    
    # Weighted Average
    wa_result = fusion.weighted_average_fusion(test_scores)
    print(f"\n1. Weighted Average:")
    print(f"   Final Score: {wa_result['final_score']:.3f}")
    print(f"   Allowed: {wa_result['all_thresholds_passed']}")
    
    # Rule-Based
    rb_result = fusion.rule_based_fusion(test_scores, test_context)
    print(f"\n2. Rule-Based:")
    print(f"   Final Score: {rb_result['final_score']:.3f}")
    print(f"   Rules Passed: {len(rb_result['rules_passed'])}/{len(rb_result['rules_passed']) + len(rb_result['rules_failed'])}")
    
    # Machine Learning (if trained)
    ml_result = fusion.machine_learning_fusion(test_scores, test_context)
    print(f"\n3. Machine Learning ({ml_result['model_used']}):")
    print(f"   Final Score: {ml_result['final_score']:.3f}")
    print(f"   Confidence: {ml_result['confidence']:.3f}")
    
    # Ensemble
    ensemble_result = fusion.ensemble_fusion(test_scores, test_context)
    print(f"\n4. Ensemble:")
    print(f"   Final Score: {ensemble_result['final_score']:.3f}")
    print(f"   Confidence: {ensemble_result['confidence']:.3f}")
    
    # Adaptive
    adaptive_result = fusion.adaptive_fusion(test_scores, test_context)
    print(f"\n5. Adaptive:")
    print(f"   Final Score: {adaptive_result['final_score']:.3f}")
    print(f"   Method: {adaptive_result['method']}")
    
    print(f"\n[CHART] Fusion System Info:")
    print(json.dumps(fusion.get_model_info(), indent=2))