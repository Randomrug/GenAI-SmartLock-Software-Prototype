"""
Behavioral Model Module
Models and predicts normal user behavior patterns
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import joblib
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class BehaviorModel:
    """
    Behavioral modeling system for user access patterns
    Learns normal behavior and detects deviations
    """
    
    def __init__(self, model_dir: str = 'models/behavior_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Behavioral models
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.behavior_profiles = {}
        
        # Owner-specific schedules and patterns
        self.owner_patterns = {
            'rithika': {
                'name': 'Rithika',
                'institution': 'College',
                'typical_hours': list(range(8, 20)),  # 8 AM to 8 PM
                'school_start': 8,
                'school_end': 19,  # 7 PM max
                'lunch_return_hour': 13,
                'lunch_return_minute': 10,
                'lunch_window': (12, 14),  # 12 PM to 2 PM
                'expected_daily_accesses': 2,  # Usually: morning arrival + lunch return
            },
            'sid': {
                'name': 'Sid',
                'institution': 'School',
                'typical_hours': list(range(8, 17)),  # 8 AM to 5 PM
                'school_start': 8,
                'school_end': 16,  # 4 PM
                'expected_daily_accesses': 1,  # Usually: arrival
            }
        }
        
        # Configuration
        self.config = {
            'clustering': {
                'n_clusters': 3,
                'min_samples_for_clustering': 50,
                'update_interval_days': 7
            },
            'profiling': {
                'min_samples_per_profile': 10,
                'profile_confidence_threshold': 0.7,
                'max_profiles_per_user': 5
            },
            'seasonality': {
                'detect_daily_patterns': True,
                'detect_weekly_patterns': True,
                'detect_monthly_patterns': False
            }
        }
        
        # Load existing models
        self.load_models()
    
    def extract_behavioral_features(self, events: List[Dict], user_id: str = 'default') -> pd.DataFrame:
        """
        Extract behavioral features from access events
        
        Args:
            events: List of access events
            user_id: User identifier for personalized modeling
            
        Returns:
            DataFrame with behavioral features
        """
        if not events:
            return pd.DataFrame()
        
        features_list = []
        
        for i, event in enumerate(events):
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(event['record_created_at'].replace('Z', '+00:00'))
                
                # Time-based features
                hour = timestamp.hour
                minute = timestamp.minute
                day_of_week = timestamp.weekday()
                day_of_month = timestamp.day
                month = timestamp.month
                is_weekend = 1 if day_of_week >= 5 else 0
                
                # Access pattern features
                access_type = 1 if event.get('action') == 'UNLOCK' else 0
                success = 1 if event.get('genai_decision') == 'ALLOW' else 0
                
                # Biometric consistency features
                face_score = event.get('face_score', 0.5)
                voice_score = event.get('voice_score', 0.5)
                final_score = event.get('final_score', 0.5)
                pin_valid = 1 if event.get('pin_valid') else 0
                
                # Risk and failure patterns
                risk_level = event.get('genai_risk_level', 'MEDIUM')
                risk_numeric = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}.get(risk_level, 1)
                failed_attempts = event.get('failed_attempt_count', 0)
                
                # Session duration (if available)
                session_duration = 0
                if event.get('entry_time') and event.get('exit_time'):
                    try:
                        entry = datetime.fromisoformat(event['entry_time'].replace('Z', '+00:00'))
                        exit = datetime.fromisoformat(event['exit_time'].replace('Z', '+00:00'))
                        session_duration = (exit - entry).total_seconds() / 60  # minutes
                    except:
                        pass
                
                # Time since last access
                time_since_last = 0
                if i > 0:
                    prev_time = datetime.fromisoformat(
                        events[i-1]['record_created_at'].replace('Z', '+00:00')
                    )
                    time_since_last = (timestamp - prev_time).total_seconds() / 3600  # hours
                
                # Create comprehensive feature vector
                features = {
                    # Temporal features
                    'hour': hour,
                    'minute': minute,
                    'day_of_week': day_of_week,
                    'day_of_month': day_of_month,
                    'month': month,
                    'is_weekend': is_weekend,
                    
                    # Circular encoding for time
                    'hour_sin': np.sin(2 * np.pi * hour / 24),
                    'hour_cos': np.cos(2 * np.pi * hour / 24),
                    'day_sin': np.sin(2 * np.pi * day_of_week / 7),
                    'day_cos': np.cos(2 * np.pi * day_of_week / 7),
                    
                    # Access pattern
                    'access_type': access_type,
                    'success': success,
                    'session_duration': session_duration,
                    'time_since_last_access': time_since_last,
                    
                    # Biometric features
                    'face_score': face_score,
                    'voice_score': voice_score,
                    'final_score': final_score,
                    'pin_valid': pin_valid,
                    'score_variance': abs(face_score - voice_score),
                    'total_biometric_score': (face_score + voice_score) / 2,
                    
                    # Risk features
                    'risk_level': risk_numeric,
                    'failed_attempts': failed_attempts,
                    'is_high_risk': 1 if risk_level == 'HIGH' else 0,
                    
                    # Derived features
                    'time_of_day_category': self._categorize_time(hour),
                    'access_frequency_category': self._categorize_frequency(time_since_last),
                    'biometric_consistency': 1 - abs(face_score - voice_score),
                    
                    # User context
                    'user_id_hash': hash(user_id) % 1000  # Pseudonymized user ID
                }
                
                features_list.append(features)
                
            except Exception as e:
                print(f"Error extracting behavioral features: {e}")
                continue
        
        return pd.DataFrame(features_list)
    
    def _categorize_time(self, hour: int) -> int:
        """Categorize time of day"""
        if 5 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 17:
            return 1  # Afternoon
        elif 17 <= hour < 22:
            return 2  # Evening
        else:
            return 3  # Night
    
    def _categorize_frequency(self, hours_since_last: float) -> int:
        """Categorize access frequency"""
        if hours_since_last < 1:
            return 0  # Very frequent (< 1 hour)
        elif hours_since_last < 6:
            return 1  # Frequent (1-6 hours)
        elif hours_since_last < 24:
            return 2  # Daily (6-24 hours)
        else:
            return 3  # Infrequent (> 24 hours)
    
    def train_behavior_model(self, historical_events: List[Dict], user_id: str = 'default'):
        """
        Train behavioral model for user
        
        Args:
            historical_events: User's historical access events
            user_id: User identifier
        """
        print(f"Training behavioral model for user {user_id}...")
        
        # Extract features
        X = self.extract_behavioral_features(historical_events, user_id)
        
        if X.empty or len(X) < self.config['clustering']['min_samples_for_clustering']:
            print(f"[WARNING]  Insufficient data for {user_id} (need {self.config['clustering']['min_samples_for_clustering']}, have {len(X)})")
            return
        
        print(f"  Training with {len(X)} samples, {X.shape[1]} features")
        
        # Prepare data
        X_numeric = X.select_dtypes(include=[np.number])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        # Determine optimal number of clusters
        optimal_clusters = self._find_optimal_clusters(X_scaled)
        
        # Train clustering model
        clustering_model = GaussianMixture(
            n_components=optimal_clusters,
            covariance_type='full',
            random_state=42
        )
        clustering_model.fit(X_scaled)
        
        # Assign clusters
        clusters = clustering_model.predict(X_scaled)
        X['cluster'] = clusters
        
        # Create behavior profiles from clusters
        profiles = self._create_behavior_profiles(X, clusters, user_id)
        
        # Store models
        self.models[user_id] = {
            'clustering': clustering_model,
            'pca': None,  # Could add PCA for visualization
            'last_trained': datetime.now().isoformat(),
            'sample_count': len(X)
        }
        
        self.scalers[user_id] = scaler
        self.behavior_profiles[user_id] = profiles
        
        # Save models
        self.save_models(user_id)
        
        print(f"[OK] Behavioral model trained for {user_id}")
        print(f"   - Clusters: {optimal_clusters}")
        print(f"   - Profiles: {len(profiles)}")
    
    def _find_optimal_clusters(self, X_scaled: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using silhouette score"""
        if len(X_scaled) < 10:
            return min(3, len(X_scaled))
        
        best_score = -1
        best_k = 2
        
        for k in range(2, min(max_clusters, len(X_scaled) // 2) + 1):
            try:
                # Use KMeans for silhouette scoring
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
            except:
                continue
        
        return best_k
    
    def _create_behavior_profiles(self, X: pd.DataFrame, clusters: np.ndarray, user_id: str) -> Dict:
        """Create behavior profiles from clusters"""
        profiles = {}
        
        for cluster_id in np.unique(clusters):
            cluster_data = X[X['cluster'] == cluster_id]
            
            if len(cluster_data) < self.config['profiling']['min_samples_per_profile']:
                continue
            
            # Calculate profile statistics
            profile = {
                'cluster_id': int(cluster_id),
                'sample_count': int(len(cluster_data)),
                'confidence': min(1.0, len(cluster_data) / 100),  # More samples = higher confidence
                
                # Temporal patterns
                'typical_hours': self._calculate_typical_range(cluster_data['hour']),
                'typical_days': self._calculate_typical_range(cluster_data['day_of_week']),
                'typical_access_type': cluster_data['access_type'].mode().iloc[0] if not cluster_data['access_type'].mode().empty else 0,
                
                # Biometric patterns
                'avg_face_score': float(cluster_data['face_score'].mean()),
                'avg_voice_score': float(cluster_data['voice_score'].mean()),
                'avg_final_score': float(cluster_data['final_score'].mean()),
                'typical_pin_valid': cluster_data['pin_valid'].mode().iloc[0] if not cluster_data['pin_valid'].mode().empty else 1,
                
                # Risk patterns
                'avg_risk_level': float(cluster_data['risk_level'].mean()),
                'success_rate': float(cluster_data['success'].mean()),
                
                # Derived patterns
                'is_consistent': float(cluster_data['biometric_consistency'].mean()) > 0.8,
                'access_frequency': self._calculate_access_frequency(cluster_data['time_since_last_access']),
                
                # Last update
                'last_updated': datetime.now().isoformat()
            }
            
            profiles[cluster_id] = profile
        
        return profiles
    
    def _calculate_typical_range(self, series: pd.Series) -> Dict:
        """Calculate typical range for a feature"""
        if len(series) < 3:
            return {'min': float(series.min()), 'max': float(series.max())}
        
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        return {
            'min': float(max(series.min(), q1 - 1.5 * iqr)),
            'max': float(min(series.max(), q3 + 1.5 * iqr)),
            'mean': float(series.mean()),
            'std': float(series.std())
        }
    
    def _calculate_access_frequency(self, time_diffs: pd.Series) -> Dict:
        """Calculate access frequency pattern"""
        if len(time_diffs) < 2:
            return {'avg_hours': 24.0, 'consistency': 0.0}
        
        avg_hours = float(time_diffs.mean())
        std_hours = float(time_diffs.std())
        
        # Consistency score (lower std = more consistent)
        consistency = 1.0 / (1.0 + std_hours) if std_hours > 0 else 1.0
        
        return {
            'avg_hours': avg_hours,
            'std_hours': std_hours,
            'consistency': min(1.0, consistency)
        }
    
    def analyze_behavior(self, current_event: Dict, historical_context: List[Dict] = None, 
                        user_id: str = 'default') -> Dict:
        """
        Analyze current behavior against learned patterns
        
        Args:
            current_event: Current access attempt
            historical_context: Recent historical events
            user_id: User identifier
            
        Returns:
            Behavioral analysis results
        """
        # Check if model exists for user
        if user_id not in self.models:
            return self._analyze_without_model(current_event, historical_context)
        
        # Prepare data
        if historical_context:
            events_for_analysis = historical_context + [current_event]
        else:
            events_for_analysis = [current_event]
        
        # Extract features
        X = self.extract_behavioral_features(events_for_analysis, user_id)
        
        if X.empty:
            return self._create_default_analysis()
        
        # Get features for current event
        current_features = X.iloc[-1:].select_dtypes(include=[np.number])
        
        # Scale features
        scaler = self.scalers.get(user_id)
        if scaler is None:
            return self._create_default_analysis()
        
        X_scaled = scaler.transform(current_features)
        
        # Get model predictions
        model = self.models[user_id]['clustering']
        profiles = self.behavior_profiles.get(user_id, {})
        
        try:
            # Predict cluster
            cluster_probs = model.predict_proba(X_scaled)[0]
            predicted_cluster = np.argmax(cluster_probs)
            cluster_confidence = cluster_probs[predicted_cluster]
            
            # Get matching profile
            matching_profile = profiles.get(predicted_cluster, {})
            
            # Calculate deviation from profile
            deviation_score = self._calculate_deviation_score(
                current_features.iloc[0], matching_profile
            )
            
            # Calculate behavioral consistency
            consistency_score = self._calculate_consistency_score(
                events_for_analysis, user_id
            )
            
            # Create analysis results
            results = {
                'behavior_match': {
                    'predicted_cluster': int(predicted_cluster),
                    'cluster_confidence': float(cluster_confidence),
                    'matching_profile': bool(matching_profile),
                    'deviation_score': float(deviation_score),
                    'is_normal_behavior': deviation_score < 0.3 and cluster_confidence > 0.5
                },
                'temporal_analysis': self._analyze_temporal_pattern(current_event, profiles),
                'biometric_analysis': self._analyze_biometric_pattern(current_event, matching_profile),
                'risk_analysis': self._analyze_risk_pattern(current_event, historical_context),
                'consistency_score': float(consistency_score),
                'anomaly_flags': [],
                'recommendations': []
            }
            
            # Check for anomalies
            if deviation_score > 0.5:
                results['anomaly_flags'].append('High deviation from normal behavior')
            
            if cluster_confidence < 0.3:
                results['anomaly_flags'].append('Low confidence in behavior classification')
            
            if consistency_score < 0.4:
                results['anomaly_flags'].append('Low behavioral consistency')
            
            # Generate recommendations
            if results['anomaly_flags']:
                results['recommendations'].append('Consider additional verification')
                results['recommendations'].append('Review recent access patterns')
            
            return results
            
        except Exception as e:
            print(f"Error in behavior analysis: {e}")
            return self._create_default_analysis()
    
    def recognize_owner_specific_patterns(self, owner: str, current_time: datetime, action: str) -> Dict:
        """
        Recognize owner-specific behavioral patterns
        
        Args:
            owner: Owner name (rithika, sid, etc.)
            current_time: Current time of access
            action: Access action (UNLOCK, LOCK)
            
        Returns:
            Pattern recognition result
        """
        pattern_info = {
            'recognized_owner': owner,
            'is_expected_time': False,
            'is_expected_action': False,
            'pattern_type': 'UNKNOWN',
            'explanation': '',
            'confidence': 0.0
        }
        
        owner_lower = owner.lower()
        
        if owner_lower not in self.owner_patterns:
            return pattern_info
        
        schedule = self.owner_patterns[owner_lower]
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Check if time is within typical hours
        if current_hour in schedule['typical_hours']:
            pattern_info['is_expected_time'] = True
            pattern_info['confidence'] += 0.4
        
        # Owner-specific patterns for Rithika
        if owner_lower == 'rithika':
            lunch_hour = schedule['lunch_return_hour']
            lunch_window = schedule['lunch_window']
            
            # Check for lunch return pattern
            if lunch_window[0] <= current_hour <= lunch_window[1]:
                if action == 'UNLOCK':
                    pattern_info['is_expected_action'] = True
                    pattern_info['pattern_type'] = 'LUNCH_RETURN'
                    pattern_info['explanation'] = f"Rithika typically returns for lunch around {lunch_hour}:00-{lunch_hour}:30"
                    pattern_info['confidence'] += 0.3
                    
                    # Extra confidence if time is close to expected lunch time
                    if current_hour == lunch_hour and abs(current_minute - schedule['lunch_return_minute']) < 30:
                        pattern_info['confidence'] += 0.2
        
        # Owner-specific patterns for Sid
        elif owner_lower == 'sid':
            if action == 'UNLOCK' and current_hour in [8, 9]:
                pattern_info['pattern_type'] = 'SCHOOL_ARRIVAL'
                pattern_info['explanation'] = f"Sid typically arrives at school around 8:30 AM"
                pattern_info['is_expected_action'] = True
                pattern_info['confidence'] += 0.3
            elif action == 'LOCK' and current_hour in [16, 17]:
                pattern_info['pattern_type'] = 'SCHOOL_DEPARTURE'
                pattern_info['explanation'] = f"Sid typically leaves school around 4:00 PM"
                pattern_info['is_expected_action'] = True
                pattern_info['confidence'] += 0.3
        
        # Normalize confidence
        pattern_info['confidence'] = min(1.0, pattern_info['confidence'])
        
        return pattern_info
    
    def _calculate_deviation_score(self, current_features: pd.Series, profile: Dict) -> float:
        """Calculate deviation from behavior profile"""
        if not profile:
            return 0.5  # Moderate uncertainty
        
        deviations = []
        
        # Check temporal deviation
        if 'typical_hours' in profile:
            hour = current_features.get('hour', 12)
            typical_min = profile['typical_hours'].get('min', 0)
            typical_max = profile['typical_hours'].get('max', 23)
            
            if hour < typical_min or hour > typical_max:
                hour_dev = min(abs(hour - typical_min), abs(hour - typical_max)) / 24
                deviations.append(hour_dev)
        
        # Check biometric deviation
        if 'avg_face_score' in profile:
            face_score = current_features.get('face_score', 0.5)
            avg_face = profile['avg_face_score']
            face_dev = abs(face_score - avg_face) / max(avg_face, 0.1)
            deviations.append(min(face_dev, 1.0))
        
        if 'avg_voice_score' in profile:
            voice_score = current_features.get('voice_score', 0.5)
            avg_voice = profile['avg_voice_score']
            voice_dev = abs(voice_score - avg_voice) / max(avg_voice, 0.1)
            deviations.append(min(voice_dev, 1.0))
        
        # Check access type deviation
        if 'typical_access_type' in profile:
            access_type = current_features.get('access_type', 0)
            typical_type = profile['typical_access_type']
            if access_type != typical_type:
                deviations.append(0.3)
        
        if not deviations:
            return 0.0
        
        return float(np.mean(deviations))
    
    def _calculate_consistency_score(self, events: List[Dict], user_id: str) -> float:
        """Calculate behavioral consistency score"""
        if len(events) < 3:
            return 0.5
        
        try:
            # Extract time differences
            timestamps = []
            for event in events[-10:]:  # Last 10 events
                try:
                    ts = datetime.fromisoformat(event['record_created_at'].replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    continue
            
            if len(timestamps) < 3:
                return 0.5
            
            # Calculate time differences
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
                time_diffs.append(diff)
            
            # Consistency: lower std = more consistent
            if len(time_diffs) >= 2:
                std = np.std(time_diffs)
                avg = np.mean(time_diffs)
                
                if avg > 0:
                    cv = std / avg  # Coefficient of variation
                    consistency = 1.0 / (1.0 + cv)
                    return float(min(consistency, 1.0))
            
        except Exception as e:
            print(f"Error calculating consistency: {e}")
        
        return 0.5
    
    def _analyze_temporal_pattern(self, event: Dict, profiles: Dict) -> Dict:
        """Analyze temporal pattern of access"""
        try:
            timestamp = datetime.fromisoformat(event['record_created_at'].replace('Z', '+00:00'))
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            # Check against typical patterns in profiles
            is_typical_hour = False
            is_typical_day = False
            
            for profile in profiles.values():
                if 'typical_hours' in profile:
                    typical_min = profile['typical_hours'].get('min', 0)
                    typical_max = profile['typical_hours'].get('max', 23)
                    
                    if typical_min <= hour <= typical_max:
                        is_typical_hour = True
                
                if 'typical_days' in profile:
                    typical_min = profile['typical_days'].get('min', 0)
                    typical_max = profile['typical_days'].get('max', 6)
                    
                    if typical_min <= day_of_week <= typical_max:
                        is_typical_day = True
            
            return {
                'hour': hour,
                'day_of_week': day_of_week,
                'is_typical_hour': is_typical_hour,
                'is_typical_day': is_typical_day,
                'time_category': self._categorize_time(hour)
            }
            
        except Exception as e:
            print(f"Error in temporal analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_biometric_pattern(self, event: Dict, profile: Dict) -> Dict:
        """Analyze biometric pattern"""
        face_score = event.get('face_score', 0.5)
        voice_score = event.get('voice_score', 0.5)
        final_score = event.get('final_score', 0.5)
        pin_valid = event.get('pin_valid', False)
        
        analysis = {
            'face_score': float(face_score),
            'voice_score': float(voice_score),
            'final_score': float(final_score),
            'pin_valid': bool(pin_valid),
            'score_consistency': 1 - abs(face_score - voice_score),
            'matches_profile': False,
            'deviation_from_profile': 0.0
        }
        
        if profile:
            avg_face = profile.get('avg_face_score', 0.5)
            avg_voice = profile.get('avg_voice_score', 0.5)
            
            face_dev = abs(face_score - avg_face) / max(avg_face, 0.1)
            voice_dev = abs(voice_score - avg_voice) / max(avg_voice, 0.1)
            
            analysis['deviation_from_profile'] = float((face_dev + voice_dev) / 2)
            analysis['matches_profile'] = analysis['deviation_from_profile'] < 0.3
        
        return analysis
    
    def _analyze_risk_pattern(self, event: Dict, historical_context: List[Dict]) -> Dict:
        """Analyze risk pattern"""
        risk_level = event.get('genai_risk_level', 'MEDIUM')
        failed_attempts = event.get('failed_attempt_count', 0)
        decision = event.get('genai_decision', 'DENY')
        
        analysis = {
            'current_risk': risk_level,
            'failed_attempts': failed_attempts,
            'decision': decision,
            'is_escalating': False,
            'recent_failures': 0
        }
        
        if historical_context:
            # Count recent failures
            recent_failures = sum(1 for e in historical_context[-5:] 
                                if e.get('genai_decision') == 'DENY')
            analysis['recent_failures'] = recent_failures
            
            # Check for escalation
            if recent_failures >= 2 and failed_attempts > 0:
                analysis['is_escalating'] = True
        
        return analysis
    
    def _analyze_without_model(self, current_event: Dict, historical_context: List[Dict]) -> Dict:
        """Analyze behavior without trained model"""
        return {
            'behavior_match': {
                'predicted_cluster': -1,
                'cluster_confidence': 0.0,
                'matching_profile': False,
                'deviation_score': 0.5,
                'is_normal_behavior': None,
                'note': 'No behavioral model available'
            },
            'temporal_analysis': self._analyze_temporal_pattern(current_event, {}),
            'biometric_analysis': self._analyze_biometric_pattern(current_event, {}),
            'risk_analysis': self._analyze_risk_pattern(current_event, historical_context),
            'consistency_score': 0.5,
            'anomaly_flags': ['Behavioral model not trained'],
            'recommendations': ['Collect more data for behavioral modeling']
        }
    
    def _create_default_analysis(self) -> Dict:
        """Create default analysis result"""
        return {
            'behavior_match': {
                'predicted_cluster': -1,
                'cluster_confidence': 0.0,
                'matching_profile': False,
                'deviation_score': 0.5,
                'is_normal_behavior': None,
                'note': 'Analysis failed'
            },
            'temporal_analysis': {},
            'biometric_analysis': {},
            'risk_analysis': {},
            'consistency_score': 0.5,
            'anomaly_flags': ['Analysis error'],
            'recommendations': []
        }
    
    def save_models(self, user_id: str = None):
        """Save models to disk"""
        try:
            if user_id and user_id in self.models:
                # Save specific user model
                user_dir = os.path.join(self.model_dir, user_id)
                os.makedirs(user_dir, exist_ok=True)
                
                model_path = os.path.join(user_dir, 'behavior_model.joblib')
                joblib.dump(self.models[user_id], model_path)
                
                if user_id in self.scalers:
                    scaler_path = os.path.join(user_dir, 'scaler.joblib')
                    joblib.dump(self.scalers[user_id], scaler_path)
                
                if user_id in self.behavior_profiles:
                    profiles_path = os.path.join(user_dir, 'profiles.json')
                    with open(profiles_path, 'w') as f:
                        json.dump(self.behavior_profiles[user_id], f, indent=2)
                
                print(f"[BACKUP] Saved model for user {user_id}")
                
            else:
                # Save all models
                for uid in self.models.keys():
                    self.save_models(uid)
        
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self):
        """Load models from disk"""
        try:
            if not os.path.exists(self.model_dir):
                return
            
            # Load user models
            for user_dir in os.listdir(self.model_dir):
                user_path = os.path.join(self.model_dir, user_dir)
                
                if os.path.isdir(user_path):
                    # Load model
                    model_path = os.path.join(user_path, 'behavior_model.joblib')
                    if os.path.exists(model_path):
                        self.models[user_dir] = joblib.load(model_path)
                    
                    # Load scaler
                    scaler_path = os.path.join(user_path, 'scaler.joblib')
                    if os.path.exists(scaler_path):
                        self.scalers[user_dir] = joblib.load(scaler_path)
                    
                    # Load profiles
                    profiles_path = os.path.join(user_path, 'profiles.json')
                    if os.path.exists(profiles_path):
                        with open(profiles_path, 'r') as f:
                            self.behavior_profiles[user_dir] = json.load(f)
            
            if self.models:
                print(f"[FOLDER] Loaded behavioral models for {len(self.models)} users")
        
        except Exception as e:
            print(f"[WARNING]  Could not load behavioral models: {e}")
    
    def get_model_info(self, user_id: str = None) -> Dict:
        """Get information about behavioral models"""
        if user_id:
            if user_id in self.models:
                model = self.models[user_id]
                profiles = self.behavior_profiles.get(user_id, {})
                
                return {
                    'user_id': user_id,
                    'last_trained': model.get('last_trained', 'Unknown'),
                    'sample_count': model.get('sample_count', 0),
                    'profile_count': len(profiles),
                    'profiles': list(profiles.keys()),
                    'is_trained': True
                }
            else:
                return {
                    'user_id': user_id,
                    'is_trained': False,
                    'message': 'No behavioral model for this user'
                }
        
        else:
            return {
                'total_users': len(self.models),
                'trained_users': list(self.models.keys()),
                'profiles_per_user': {
                    uid: len(profiles) 
                    for uid, profiles in self.behavior_profiles.items()
                }
            }

# Factory function
def create_behavior_model(model_dir: str = 'models/behavior_models') -> BehaviorModel:
    """Factory function to create behavior model"""
    return BehaviorModel(model_dir)

if __name__ == "__main__":
    # Test the behavior model
    print("[TEST] Testing Behavior Model")
    
    # Create sample events
    sample_events = []
    base_time = datetime.now()
    
    # Create normal pattern (morning unlocks)
    for i in range(50):
        event_time = base_time - timedelta(days=i, hours=8)  # 8 AM each day
        sample_events.append({
            'record_created_at': event_time.isoformat(),
            'action': 'UNLOCK',
            'face_score': 0.85 + np.random.normal(0, 0.05),
            'voice_score': 0.82 + np.random.normal(0, 0.05),
            'final_score': 0.84 + np.random.normal(0, 0.04),
            'pin_valid': True,
            'failed_attempt_count': 0,
            'genai_risk_level': 'LOW',
            'genai_decision': 'ALLOW'
        })
    
    # Create different pattern (evening locks)
    for i in range(30):
        event_time = base_time - timedelta(days=i, hours=20)  # 8 PM each day
        sample_events.append({
            'record_created_at': event_time.isoformat(),
            'action': 'LOCK',
            'face_score': 0.88 + np.random.normal(0, 0.05),
            'voice_score': 0.85 + np.random.normal(0, 0.05),
            'final_score': 0.86 + np.random.normal(0, 0.04),
            'pin_valid': True,
            'failed_attempt_count': 0,
            'genai_risk_level': 'LOW',
            'genai_decision': 'ALLOW'
        })
    
    # Test behavior model
    model = BehaviorModel()
    
    # Train model
    model.train_behavior_model(sample_events, 'test_user')
    
    # Test analysis
    test_event = {
        'record_created_at': datetime.now().isoformat(),
        'action': 'UNLOCK',
        'face_score': 0.87,
        'voice_score': 0.83,
        'final_score': 0.85,
        'pin_valid': True,
        'failed_attempt_count': 0,
        'genai_risk_level': 'LOW',
        'genai_decision': 'ALLOW'
    }
    
    analysis = model.analyze_behavior(test_event, sample_events[:20], 'test_user')
    
    print("\n[SEARCH] Behavioral Analysis Result:")
    print(json.dumps(analysis, indent=2))
    
    print(f"\n[CHART] Model Info:")
    print(json.dumps(model.get_model_info('test_user'), indent=2))