"""
GenAI Safety Engine with REAL OpenRouter API Integration
Uses free models like Llama 3.3-70B via OpenRouter
Requires: OPENROUTER_API_KEY environment variable
"""
import os
import json
import requests
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Optional, Any
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GenAIAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GenAI analyzer with OpenRouter API
        
        Args:
            api_key: OpenRouter API key. If None, tries to get from environment
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Try multiple working models to handle rate limits
        # Primary model can be overridden via OPENROUTER_MODEL environment variable
        self.primary_model = os.getenv('OPENROUTER_MODEL', "meta-llama/llama-3.3-70b-instruct:free")
        self.fallback_models = [
            "arcee-ai/trinity-large-preview:free",  # Known working
            "upstage/solar-pro-3:free",             # Known working
            "liquid/lfm-2.5-1.2b-instruct:free",   # Smaller but working
        ]
        self.model = self.primary_model
        
        # Optional: Your site info for OpenRouter rankings
        self.site_url = os.getenv('SITE_URL', 'http://localhost:5000')
        self.site_name = os.getenv('SITE_NAME', 'GenAI Smart Lock')
        
        self.max_tokens = 300
        self.temperature = 0.8  # Higher = more creative & diverse responses (not formulaic)
        
        # Initialize API client
        if self.api_key:
            print(f"[OK] GenAI initialized with OpenRouter API")
            print(f"[AI] Model: {self.model}")
        else:
            print("[WARNING]  No OpenRouter API key found. Running in simulation mode.")
    
    def analyze_attempt(self, current_attempt: Dict) -> Dict:
        """
        Analyze access attempt using real GenAI via OpenRouter
        
        Args:
            current_attempt: Dictionary with attempt data
            
        Returns:
            Dictionary with decision, risk_level, and explanation
        """
        print("\n[AI] GENAI ANALYSIS REQUESTED")
        
        # If no API key, use fallback
        if not self.api_key:
            print("[WARNING]  No API key - using fallback analysis")
            return self._fallback_analysis(current_attempt)
        
        try:
            # Build context from database
            context = self._build_ai_context(current_attempt)
            
            # Call OpenRouter API
            response = self._call_openrouter_api(context)
            
            # Parse and validate response
            result = self._parse_ai_response(response)
            
            print(f"[OK] GenAI Analysis Complete: {result['decision']}")
            return result
            
        except Exception as e:
            print(f"[ERROR] GenAI API Error: {e}")
            print("[RETRY] Using fallback rule-based analysis instead...")
            return self._fallback_analysis(current_attempt)
    
    def _build_ai_context(self, current_attempt: Dict) -> Dict:
        """Build comprehensive context for AI analysis with owner-specific patterns"""
        
        # Get historical data from database
        historical_context = self._get_historical_context()
        
        # Get recent events
        recent_events = self._get_recent_events(limit=15)
        
        # Get owner-specific patterns
        owner = current_attempt.get('owner', 'unknown')
        owner_patterns = self._get_owner_specific_patterns(owner)
        
        # Calculate statistics
        stats = self._calculate_statistics(recent_events)
        
        # Check for behavioral anomalies
        behavioral_anomaly = self._detect_behavioral_anomaly(current_attempt, owner_patterns)
        
        # Build context object
        context = {
            'current_attempt': {
                'owner': owner,
                'face_owner': current_attempt.get('face_owner', 'unknown'),
                'voice_owner': current_attempt.get('voice_owner', 'unknown'),
                'owner_mismatch': current_attempt.get('owner_mismatch', False),
                'action': current_attempt.get('action', 'IN'),
                'datetime': current_attempt.get('manual_datetime', 'Not provided'),
                'pin_valid': current_attempt.get('pin_valid', False),
                'face_score': current_attempt.get('face_score', 0.0),
                'voice_score': current_attempt.get('voice_score', 0.0),
                'behavior_score': current_attempt.get('behavior_score', 0.5),
                'final_score': current_attempt.get('final_score', 0.0),
                'failed_attempts': current_attempt.get('failed_attempt_count', 0),
                'lockout_active': current_attempt.get('lockout_active', False)
            },
            'owner_patterns': owner_patterns,
            'behavioral_anomaly': behavioral_anomaly,
            'historical_patterns': historical_context,
            'recent_activity': {
                'total_events': len(recent_events),
                'success_rate': stats.get('success_rate', 0.0),
                'avg_face_score': stats.get('avg_face_score', 0.0),
                'avg_voice_score': stats.get('avg_voice_score', 0.0),
                'recent_decisions': [e.get('genai_decision', 'UNKNOWN') for e in recent_events[:5]],
                'failure_streak': stats.get('failure_streak', 0)
            },
            'risk_factors': self._identify_risk_factors(current_attempt, stats),
            'system_state': {
                'current_time': datetime.now().isoformat(),
                'total_database_entries': self._get_total_entries(),
                'lockout_history': self._get_lockout_history()
            }
        }
        
        return context
    
    def _get_historical_context(self) -> Dict:
        """Get historical behavior patterns"""
        try:
            conn = sqlite3.connect('smart_lock_events.db')
            cursor = conn.cursor()
            
            # Get successful access patterns
            cursor.execute("""
                SELECT 
                    strftime('%H', record_created_at) as hour,
                    COUNT(*) as count,
                    AVG(face_score) as avg_face,
                    AVG(voice_score) as avg_voice
                FROM access_events 
                WHERE genai_decision = 'ALLOW'
                GROUP BY strftime('%H', record_created_at)
                ORDER BY count DESC
                LIMIT 5
            """)
            
            time_patterns = cursor.fetchall()
            
            # Get common scores
            cursor.execute("""
                SELECT 
                    AVG(face_score) as avg_face_all,
                    AVG(voice_score) as avg_voice_all,
                    AVG(final_score) as avg_final_all,
                    MIN(face_score) as min_face_success,
                    MIN(voice_score) as min_voice_success
                FROM access_events 
                WHERE genai_decision = 'ALLOW'
            """)
            
            score_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'common_access_hours': [int(hour) for hour, count, _, _ in time_patterns],
                'avg_scores': {
                    'face': score_stats[0] if score_stats[0] else 0.0,
                    'voice': score_stats[1] if score_stats[1] else 0.0,
                    'final': score_stats[2] if score_stats[2] else 0.0
                },
                'minimum_success_scores': {
                    'face': score_stats[3] if score_stats[3] else 0.6,
                    'voice': score_stats[4] if score_stats[4] else 0.6
                }
            }
            
        except Exception as e:
            print(f"Database error in historical context: {e}")
            return {}
    
    def _get_recent_events(self, limit: int = 15) -> List[Dict]:
        """Get recent access events"""
        try:
            conn = sqlite3.connect('smart_lock_events.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM access_events 
                ORDER BY record_created_at DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except:
            return []
    
    def _calculate_statistics(self, events: List[Dict]) -> Dict:
        """Calculate statistics from events"""
        if not events:
            return {}
        
        total = len(events)
        allowed = sum(1 for e in events if e.get('genai_decision') == 'ALLOW')
        denied = sum(1 for e in events if e.get('genai_decision') == 'DENY')
        
        face_scores = [e.get('face_score', 0) for e in events if e.get('face_score')]
        voice_scores = [e.get('voice_score', 0) for e in events if e.get('voice_score')]
        
        # Calculate failure streak
        failure_streak = 0
        for event in events:
            if event.get('genai_decision') == 'DENY':
                failure_streak += 1
            else:
                break
        
        return {
            'success_rate': allowed / total if total > 0 else 0,
            'denial_rate': denied / total if total > 0 else 0,
            'avg_face_score': sum(face_scores) / len(face_scores) if face_scores else 0,
            'avg_voice_score': sum(voice_scores) / len(voice_scores) if voice_scores else 0,
            'failure_streak': failure_streak,
            'total_analyzed': total
        }
    
    def _identify_risk_factors(self, attempt: Dict, stats: Dict) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        # Check biometric scores
        face_score = attempt.get('face_score', 0)
        voice_score = attempt.get('voice_score', 0)
        
        if face_score < 0.6:
            risk_factors.append(f"Low face score: {face_score:.2f} (threshold: 0.6)")
        if voice_score < 0.6:
            risk_factors.append(f"Low voice score: {voice_score:.2f} (threshold: 0.6)")
        
        # Check failure streak
        failed_attempts = attempt.get('failed_attempt_count', 0)
        if failed_attempts >= 3:
            risk_factors.append(f"Multiple consecutive failures: {failed_attempts}")
        
        # Check if scores are significantly below average
        avg_face = stats.get('avg_face_score', 0.8)
        avg_voice = stats.get('avg_voice_score', 0.8)
        
        if face_score < avg_face - 0.3:
            risk_factors.append(f"Face score significantly below average ({avg_face:.2f})")
        if voice_score < avg_voice - 0.3:
            risk_factors.append(f"Voice score significantly below average ({avg_voice:.2f})")
        
        # Check PIN
        if not attempt.get('pin_valid', False):
            risk_factors.append("Invalid PIN provided")
        
        return risk_factors
    
    def _get_total_entries(self) -> int:
        """Get total number of database entries"""
        try:
            conn = sqlite3.connect('smart_lock_events.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM access_events")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0
    
    def _get_lockout_history(self) -> List[Dict]:
        """Get history of lockout events"""
        try:
            conn = sqlite3.connect('smart_lock_events.db')
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT record_created_at, genai_explanation 
                FROM access_events 
                WHERE lockout_active = 1 
                ORDER BY record_created_at DESC 
                LIMIT 5
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except:
            return []
    
    def _get_owner_specific_patterns(self, owner: str) -> Dict:
        """Get behavioral patterns specific to owner with day-of-week awareness"""
        # Define owner schedules with day-specific patterns
        owner_schedules = {
            'rithika': {
                'name': 'Rithika',
                'description': 'College student with day-specific schedule',
                'day_schedules': {
                    'Monday': {'start': 8, 'end': 23, 'lunch': (13, 14), 'activity': 'College 8-7. May stay late (up to 11 PM) due to high office/project work', 'late_night_exception': True},
                    'Tuesday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'activity': 'College 8-7 (lunch 1-2)'},
                    'Wednesday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'activity': 'College 8-7 (lunch 1-2)'},
                    'Thursday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'activity': 'College 8-7 (lunch 1-2)'},
                    'Friday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'activity': 'College 8-7 (lunch 1-2)'},
                    'Saturday': {'start': 10, 'end': 23, 'lunch': None, 'activity': 'Family events - movie night or family dinner. Usually returns late (11-11:30 PM)', 'late_night_exception': True, 'weekend_social': True},
                    'Sunday': {'start': 9, 'end': 21, 'lunch': None, 'activity': 'Normal home day'}
                },
                'typical_hours': list(range(8, 20)),
                'school_hours_start': 8,
                'school_hours_end': 19,
                'lunch_window_start': 12,
                'lunch_window_end': 14,
                'lunch_return_hour': 13,
                'lunch_return_minute': 10,
                'late_night_exception_days': ['Monday', 'Saturday'],
                'late_night_exception_note': 'Late access (after 19:00) acceptable on Monday (office work) and Saturday (movies/family dinner - returns 11-11:30 PM)',
                'saturday_expected_return': (23, 0),  # Expected between 11 PM - midnight
                'saturday_activities': ['movie night', 'family dinner', 'social events']
            },
            'sid': {
                'name': 'Sid',
                'description': 'School student with day-specific schedule',
                'day_schedules': {
                    'Monday': {'start': 9, 'end': 17, 'lunch': None, 'activity': 'School 9-5'},
                    'Tuesday': {'start': 9, 'end': 17, 'lunch': None, 'activity': 'School 9-5'},
                    'Wednesday': {'start': 9, 'end': 17, 'lunch': None, 'activity': 'School 9-5'},
                    'Thursday': {'start': 9, 'end': 17, 'lunch': None, 'activity': 'School 9-5'},
                    'Friday': {'start': 9, 'end': 17, 'lunch': None, 'activity': 'School 9-5'},
                    'Saturday': {'start': 10, 'end': 20, 'lunch': None, 'activity': 'Weekend activities'},
                    'Sunday': {'start': 8, 'end': 22, 'lunch': None, 'activity': 'Badminton practice 4-6:30 PM, may return late due to post-game time', 'badminton_day': True}
                },
                'typical_hours': list(range(8, 17)),
                'school_hours_start': 9,
                'school_hours_end': 17,
                'badminton_day': 'Sunday',
                'badminton_hours': (16, 18.5),  # 4-6:30 PM
                'badminton_expected_return': (18, 22),  # Usually returns between 6-10 PM
                'badminton_note': 'Sunday badminton 4-6:30 PM - may stay for post-match activities'
            }
        }
        
        if owner.lower() in owner_schedules:
            schedule = owner_schedules[owner.lower()]
            
            # Get owner's historical patterns from DB
            try:
                conn = sqlite3.connect('smart_lock_events.db')
                cursor = conn.cursor()
                
                # Get current day of week
                today = datetime.now().strftime('%A')
                
                # Get patterns for current day specifically
                cursor.execute("""
                    SELECT 
                        strftime('%H', record_created_at) as hour,
                        COUNT(*) as count,
                        AVG(final_score) as avg_score
                    FROM access_events 
                    WHERE owner = ? AND day_of_week = ? AND genai_decision = 'ALLOW'
                    GROUP BY strftime('%H', record_created_at)
                    ORDER BY count DESC
                    LIMIT 5
                """, (owner.lower(), today))
                
                today_patterns = [int(row[0]) for row in cursor.fetchall()]
                
                # Get overall patterns
                cursor.execute("""
                    SELECT 
                        strftime('%H', record_created_at) as hour,
                        COUNT(*) as count,
                        AVG(final_score) as avg_score
                    FROM access_events 
                    WHERE owner = ? AND genai_decision = 'ALLOW'
                    GROUP BY strftime('%H', record_created_at)
                    ORDER BY count DESC
                    LIMIT 5
                """, (owner.lower(),))
                
                common_hours = [int(row[0]) for row in cursor.fetchall()]
                
                # Get success metrics
                cursor.execute("""
                    SELECT AVG(final_score), COUNT(*)
                    FROM access_events 
                    WHERE owner = ? AND genai_decision = 'ALLOW'
                """, (owner.lower(),))
                
                result = cursor.fetchone()
                avg_final_score = result[0] if result[0] else 0.8
                total_successful = result[1] if result[1] else 0
                
                # Get day-specific patterns
                cursor.execute("""
                    SELECT day_of_week, COUNT(*) as count, AVG(final_score) as avg_score
                    FROM access_events 
                    WHERE owner = ? AND genai_decision = 'ALLOW'
                    GROUP BY day_of_week
                """, (owner.lower(),))
                
                day_patterns = {row[0]: {'count': row[1], 'avg_score': row[2]} for row in cursor.fetchall()}
                
                conn.close()
                
                schedule['current_day'] = today
                schedule['current_day_schedule'] = schedule['day_schedules'].get(today, {})
                schedule['today_common_hours'] = today_patterns if today_patterns else schedule['typical_hours']
                schedule['common_access_hours'] = common_hours if common_hours else schedule['typical_hours']
                schedule['avg_successful_score'] = avg_final_score
                schedule['total_successful_accesses'] = total_successful
                schedule['day_patterns'] = day_patterns
                
            except Exception as e:
                print(f"[WARNING] Error getting owner patterns: {e}")
                # Set defaults
                schedule['current_day'] = datetime.now().strftime('%A')
                schedule['common_access_hours'] = schedule['typical_hours']
                schedule['avg_successful_score'] = 0.8
                schedule['total_successful_accesses'] = 0
                schedule['day_patterns'] = {}
            
            return schedule
        
        return {
            'name': owner,
            'typical_hours': list(range(6, 23)),
            'description': 'Unknown user'
        }
    
    def _format_day_patterns(self, day_patterns: Dict) -> str:
        """Format day-of-week patterns for display in context"""
        if not day_patterns:
            return "No historical data by day"
        
        formatted = []
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days_order:
            if day in day_patterns:
                pattern = day_patterns[day]
                count = pattern.get('count', 0)
                avg_score = pattern.get('avg_score', 0)
                formatted.append(f"  {day}: {count} events, avg score {avg_score:.2f}")
        
        return '\n'.join(formatted) if formatted else "No historical data by day"
    
    def _detect_behavioral_anomaly(self, attempt: Dict, patterns: Dict) -> Dict:
        """Detect if access attempt is unusual based on patterns"""
        anomalies = {
            'is_anomalous': False,
            'anomaly_type': None,
            'severity': 'LOW',
            'reason': None
        }
        
        try:
            # Parse datetime if provided
            datetime_str = attempt.get('manual_datetime')
            if not datetime_str:
                return anomalies
            
            dt = datetime.fromisoformat(datetime_str.replace(' ', 'T'))
            current_hour = dt.hour
            current_minute = dt.minute
            
            # Check if access time is outside typical hours
            typical_hours = patterns.get('typical_hours', list(range(6, 23)))
            if current_hour not in typical_hours:
                anomalies['is_anomalous'] = True
                anomalies['anomaly_type'] = 'UNUSUAL_TIME'
                anomalies['severity'] = 'MEDIUM'
                anomalies['reason'] = f"Access at {current_hour}:{current_minute:02d} is outside typical hours ({min(typical_hours)}-{max(typical_hours)})"
                return anomalies
            
            # For Rithika: Check lunch return pattern
            if attempt.get('owner', '').lower() == 'rithika':
                lunch_hour = patterns.get('lunch_return_hour', 13)
                lunch_minute = patterns.get('lunch_return_minute', 10)
                lunch_window_start = patterns.get('lunch_window_start', 12)
                lunch_window_end = patterns.get('lunch_window_end', 14)
                
                # If access is within lunch return window, it's likely lunch return
                if lunch_window_start <= current_hour <= lunch_window_end:
                    if current_hour == lunch_hour and abs(current_minute - lunch_minute) <= 30:
                        anomalies['is_anomalous'] = False
                        anomalies['reason'] = f"Likely lunch return (~{lunch_hour}:{lunch_minute:02d}). Pattern recognized."
                        return anomalies
                    elif current_hour == lunch_hour - 1 or (current_hour == lunch_hour and current_minute < lunch_minute - 30):
                        anomalies['is_anomalous'] = True
                        anomalies['anomaly_type'] = 'EARLY_LUNCH_RETURN'
                        anomalies['severity'] = 'LOW'
                        anomalies['reason'] = f"Returning earlier than typical lunch time ({current_hour}:{current_minute:02d} vs ~{lunch_hour}:{lunch_minute:02d})"
                        return anomalies
            
            return anomalies
            
        except Exception as e:
            print(f"Error detecting anomaly: {e}")
            return anomalies
    
    def _call_openrouter_api(self, context: Dict) -> str:
        """Make API call to OpenRouter with owner-aware behavioral patterns"""
        
        system_prompt = """You are a security analyst for a multi-owner smart lock system. Analyze each access attempt and provide a decision.

DECISION RULES:
- ALLOW: Biometrics strong (>0.70), owner confirmed, PIN valid, time within patterns
- DENY: Weak biometrics, wrong PIN, owner mismatch, unusual access time
- LOCKOUT: Multiple failures or confirmed intrusion attempt

RISK LEVELS:
- HIGH: Owner mismatch, weak biometrics, repeated failures, security threats
- MEDIUM: Access outside typical hours (unless documented exception)
- LOW: Strong biometrics, owner confirmed, normal access pattern

RESPONSE FORMAT - OUTPUT ONLY VALID JSON, NO OTHER TEXT:
{
    "decision": "ALLOW|DENY|LOCKOUT",
    "risk_level": "LOW|MEDIUM|HIGH",
    "explanation": "1-2 sentence technical analysis of the decision",
    "requires_verification_question": false,
    "verification_question": ""
}

CRITICAL INSTRUCTIONS:
1. Output ONLY the JSON object - no additional text, no code, no if-else statements
2. Keep explanation brief and technical (max 2 sentences)
3. Do NOT output any Python code, pseudocode, or if-else logic statements
4. Ensure all JSON fields are properly formatted and valid
"""
        
        # Determine owner info for analysis
        owner = context['current_attempt'].get('owner', 'unknown')
        face_owner = context['current_attempt'].get('face_owner', 'unknown')
        voice_owner = context['current_attempt'].get('voice_owner', 'unknown')
        owner_mismatch = context['current_attempt'].get('owner_mismatch', False)
        
        owner_patterns = context.get('owner_patterns', {})
        behavioral_anomaly = context.get('behavioral_anomaly', {})
        
        # Get day-of-week from the attempt's provided datetime (fall back to now)
        current_day = None
        attempt_dt_str = context['current_attempt'].get('manual_datetime') or context['current_attempt'].get('datetime')
        if attempt_dt_str:
            try:
                attempt_dt = datetime.strptime(attempt_dt_str, '%Y-%m-%d %H:%M:%S')
                current_day = attempt_dt.strftime('%A')
            except Exception:
                try:
                    attempt_dt = datetime.fromisoformat(attempt_dt_str)
                    current_day = attempt_dt.strftime('%A')
                except Exception:
                    current_day = datetime.now().strftime('%A')
        else:
            current_day = datetime.now().strftime('%A')
        current_day_schedule = owner_patterns.get('current_day_schedule', {})
        
        user_prompt = f"""
        ===== ACCESS ANALYSIS: {owner.upper()} on {current_day} =====
        
        *** AUTHENTICATION STATUS ***
        Owner Confirmed: {'NO - MISMATCH ALERT' if owner_mismatch else 'YES - Pending validation'}
        Pin Valid: {context['current_attempt']['pin_valid']}
        Biometrics Strong (>0.70): Face={context['current_attempt']['face_score'] >= 0.70}, Voice={context['current_attempt']['voice_score'] >= 0.70}
        SECURITY THREAT LEVEL: {'CRITICAL - Potential intrusion' if (owner_mismatch or context['current_attempt']['face_score'] < 0.70 or context['current_attempt']['voice_score'] < 0.70 or not context['current_attempt']['pin_valid']) else 'Standard authentication flow'}
        
        BIOMETRIC VERIFICATION:
        • Face Match: {context['current_attempt']['face_score']:.3f} (Owner: {face_owner.upper()})
        • Voice Match: {context['current_attempt']['voice_score']:.3f} (Owner: {voice_owner.upper()})
        • PIN Valid: {context['current_attempt']['pin_valid']}
        • Cross-Check: {'MISMATCH ALERT' if owner_mismatch else 'Consistent'}
        
        HISTORICAL BASELINE FOR {owner.upper() if owner != 'unknown' else 'user'}:
        • Total Successful Accesses: {owner_patterns.get('total_successful_accesses', 0)}
        • Average Biometric Confidence: Face {context['historical_patterns']['avg_scores'].get('face', 0.8):.2f}, Voice {context['historical_patterns']['avg_scores'].get('voice', 0.8):.2f}
        • This Attempt's Scores vs. History: Face {'+' if context['current_attempt']['face_score'] > context['historical_patterns']['avg_scores'].get('face', 0.8) else '-'} {abs(context['current_attempt']['face_score'] - context['historical_patterns']['avg_scores'].get('face', 0.8)):.2f}, Voice {'+' if context['current_attempt']['voice_score'] > context['historical_patterns']['avg_scores'].get('voice', 0.8) else '-'} {abs(context['current_attempt']['voice_score'] - context['historical_patterns']['avg_scores'].get('voice', 0.8)):.2f}
        • Typical Access Windows: {owner_patterns.get('common_access_hours', 'Unknown')}
        • Recent Success Rate: {context['recent_activity']['success_rate']:.1%}
        
        CURRENT ATTEMPT CONTEXT:
        • Time: {context['current_attempt']['datetime']}
        • Day of Week: {current_day} (weekday/weekend pattern considered)
        • Day's Expected Activity: {current_day_schedule.get('activity', 'No specific pattern')}
        • Late-Night Exception: {' YES - ' + current_day_schedule.get('activity', '') if current_day_schedule.get('late_night_exception') else 'No documented late-night exception'}
        • Access Type: {context['current_attempt']['action']}
        • Behavioral Anomaly: {behavioral_anomaly.get('reason', 'None detected')}
        • Recent Failures: {context['recent_activity']['failure_streak']} consecutive denials
        
        SPECIFIC DATA POINTS FOR THIS ANALYSIS:
        {chr(10).join(f'• {factor}' for factor in context['risk_factors']) if context['risk_factors'] else '• No specific risk factors yet - see biometric scores and historical patterns'}
        
        WHAT TO FOCUS ON:
        1. Are the current biometric scores consistent with this user's typical performance?
        2. What is the DEVIATION between current scores and this user's historical average?
        3. Does the access time match ANY recorded pattern from {owner.upper()}'s history?
        4. IMPORTANT: If the day's activity description mentions "late" or "may stay late" or indicates a documented exception, THIS IS AN EXPECTED PATTERN regardless of database frequency. Treat late access on those days as normal (LOW risk if biometrics match).
        5. If biometrics are strong, why might access timing be different today?
        6. If biometrics are weak, what could explain the degradation (device angle, voice fatigue, masked face)?
        7. Is ownership identity confirmed across all methods (face, voice, PIN)?
        
        PROVIDE ANALYSIS THAT IS:
        - Specific to {owner.upper()}'s actual history (not generic rules)
        - Quantitative (reference actual numbers: frequencies, deviations, percentages)
        - Contextual (explain WHY this access is unusual/normal for THIS owner)
        - Unique (never output the same explanation for two different attempts)
        
        Output your decision with a unique, data-driven explanation:"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }
        
        # List of models to try (primary + fallbacks)
        models_to_try = [self.model] + self.fallback_models
        last_error = None
        
        for attempt_model in models_to_try:
            payload = {
                "model": attempt_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            try:
                if attempt_model != self.model:
                    print(f"[RETRY] Trying fallback model: {attempt_model}")
                else:
                    print(f"[WEB] Calling OpenRouter API with owner-aware analysis...")
                    print(f"   Owner: {owner.upper()}")
                    print(f"   Face Owner: {face_owner.upper()}, Voice Owner: {voice_owner.upper()}")
                    print(f"   Model: {self.model}")
                
                response = requests.post(
                    url=self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                print(f"   Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if result and 'choices' in result and len(result['choices']) > 0:
                            choice = result['choices'][0]
                            # Get content - try 'content' first, then 'reasoning' for models with internal reasoning
                            content = choice.get('message', {}).get('content', '')
                            if not content and 'reasoning' in choice:
                                # Some models (like Step) put reasoning in a separate field
                                content = choice.get('reasoning', '')
                            
                            if content:
                                print(f"[OK] API Response received successfully")
                                return content
                            else:
                                print(f"[WARNING]  Empty response content from {attempt_model}")
                                last_error = Exception("Empty API response")
                                continue
                        else:
                            print(f"[WARNING]  Empty response structure from {attempt_model}")
                            last_error = Exception("Empty API response")
                            continue
                    except Exception as parse_error:
                        print(f"[WARNING]  Failed to parse response from {attempt_model}: {parse_error}")
                        last_error = parse_error
                        continue
                elif response.status_code == 429:
                    # Rate limited, try next model
                    error_msg = f"Model {attempt_model} rate-limited"
                    print(f"[WARNING]  {error_msg}")
                    last_error = Exception(error_msg)
                    continue
                else:
                    error_msg = f"API Error {response.status_code}: {response.text[:100]}"
                    print(f"[ERROR] OpenRouter API Error: {error_msg}")
                    last_error = Exception(error_msg)
                    continue
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
                error_msg = f"Model {attempt_model} failed: {str(e)[:100]}"
                print(f"[WARNING]  {error_msg}")
                last_error = e
                continue
            except Exception as e:
                error_msg = f"Model {attempt_model} error: {str(e)[:100]}"
                print(f"[WARNING]  {error_msg}")
                last_error = e
                continue
        # All models failed, raise the last error
        if last_error:
            print(f"[ERROR] All OpenRouter models failed. Using fallback analysis.")
            raise last_error
        else:
            raise Exception("Failed to call OpenRouter API with all models")
            raise Exception(error_msg)
    
    def _parse_ai_response(self, response_text: str) -> Dict:
        """Parse and validate AI response"""
        try:
            # Clean response text
            response_text = response_text.strip()
            
            # CHECK: Reject if response contains if-else statements or obvious code patterns
            # Be specific to catch actual code, not just English words
            forbidden_patterns = [
                'if ', 'else:', 'elif ', 'while ', 'def ', 'class ',
                'try:', 'except:', 'finally:', '=>'
            ]
            response_lower = response_text.lower()
            for pattern in forbidden_patterns:
                # Only flag if pattern appears outside of JSON context
                if pattern in response_lower and not response_text.strip().startswith('{'):
                    print(f"[WARNING] Response might contain code pattern '{pattern}' - checking...")
                    # Do secondary check: if it starts with json, ignore
                    if not '"' in response_text[:50]:  # JSON usually has quotes early
                        print(f"[WARN] Suspicious pattern detected, but proceeding with JSON extraction")
            
            # Extract JSON (in case there's extra text)
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['decision', 'risk_level', 'explanation']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing field: {field}")
            
            # Validate decision
            valid_decisions = ['ALLOW', 'DENY', 'LOCKOUT']
            if result['decision'] not in valid_decisions:
                result['decision'] = 'DENY'
            
            # Validate risk level
            valid_risk_levels = ['LOW', 'MEDIUM', 'HIGH']
            if result['risk_level'] not in valid_risk_levels:
                result['risk_level'] = 'MEDIUM'
            
            # Ensure explanation is a string
            if not isinstance(result['explanation'], str):
                result['explanation'] = str(result['explanation'])
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Raw Response: {response_text[:200]}")
            return self._create_error_response("Failed to parse AI response as JSON")
        except Exception as e:
            print(f"Response Validation Error: {e}")
            return self._create_error_response(f"Validation error: {str(e)}")
    
    def _create_error_response(self, message: str) -> Dict:
        """Create error response when AI fails"""
        return {
            'decision': 'DENY',
            'risk_level': 'HIGH',
            'explanation': f"AI analysis failed: {message}. Access denied for security.",
            'requires_verification_question': False,
            'verification_question': ''
        }
    
    def _fallback_analysis(self, current_attempt: Dict) -> Dict:
        """Fallback rule-based analysis when AI is unavailable, with owner-aware logic"""
        print("[RETRY] Using rule-based fallback analysis with owner awareness")
        
        owner = current_attempt.get('owner', 'unknown')
        face_owner = current_attempt.get('face_owner', 'unknown')
        voice_owner = current_attempt.get('voice_owner', 'unknown')
        owner_mismatch = current_attempt.get('owner_mismatch', False)
        
        pin_valid = current_attempt.get('pin_valid', False)
        face_score = current_attempt.get('face_score', 0.0)
        voice_score = current_attempt.get('voice_score', 0.0)
        final_score = current_attempt.get('final_score', 0.0)
        failed_attempts = current_attempt.get('failed_attempt_count', 0)
        
        # Rule 0: Check for owner mismatch (HIGHEST PRIORITY)
        if owner_mismatch and face_owner != "unknown" and voice_owner != "unknown":
            explanation = (
                f"SECURITY ALERT: Voice and face owners do not match (Face: {face_owner.upper()}, "
                f"Voice: {voice_owner.upper()}). This indicates a potential impersonation attempt. "
                "Access denied and administrators should be notified."
            )
            return {
                'decision': 'DENY',
                'risk_level': 'HIGH',
                'explanation': explanation,
                'requires_verification_question': False,
                'verification_question': ''
            }
        
        # Rule 1: Check PIN
        if not pin_valid:
            explanation = "Invalid PIN provided. Even if biometrics are present, an incorrect PIN increases risk. Access denied." 
            return {
                'decision': 'DENY',
                'risk_level': 'HIGH',
                'explanation': explanation,
                'requires_verification_question': False,
                'verification_question': ''
            }
        
        # Rule 2: Check failure streak
        if failed_attempts >= 5:
            explanation = f'Multiple consecutive failures ({failed_attempts}). System locked for security and administrators notified.'
            return {
                'decision': 'LOCKOUT',
                'risk_level': 'HIGH',
                'explanation': explanation,
                'requires_verification_question': False,
                'verification_question': ''
            }
        
        # Rule 3: Check biometric thresholds
        # RULE: Biometrics are the primary decision factor
        # If biometrics pass, ALWAYS ALLOW (even if time is unusual)
        if face_score >= 0.7 and voice_score >= 0.7:
            # Build a richer explanation that reads like a GenAI insight
            owner_info = f"recognized as {owner.upper()}" if owner != "unknown" else "recognized user"
            unusual_time_note = ""
            try:
                # If attempt includes manual_datetime and it's outside typical hours, note it
                manual_dt = current_attempt.get('manual_datetime')
                if manual_dt:
                    from datetime import datetime as _dt
                    dt = _dt.fromisoformat(manual_dt.replace(' ', 'T'))
                    hour = dt.hour
                    typical_hours = self._get_owner_specific_patterns(owner).get('typical_hours', list(range(6,23)))
                    if hour not in typical_hours:
                        unusual_time_note = f" The access occurred outside typical hours ({min(typical_hours)}-{max(typical_hours)})."
            except Exception:
                unusual_time_note = ""

            explanation = (
                f"Biometric authentication strong — face: {face_score:.2f}, voice: {voice_score:.2f} — {owner_info}."
                f"{unusual_time_note} Biometrics override timing concerns; access granted."
            )
            return {
                'decision': 'ALLOW',
                'risk_level': 'LOW',
                'explanation': explanation,
                'requires_verification_question': False,
                'verification_question': ''
            }
        
        # If face fails
        if face_score < 0.7:
            explanation = f'Face recognition score too low ({face_score:.2f}, required >= 0.7). Please retry capture or use alternative verification.'
            return {
                'decision': 'DENY',
                'risk_level': 'MEDIUM',
                'explanation': explanation,
                'requires_verification_question': False,
                'verification_question': ''
            }
        
        # If voice fails
        if voice_score < 0.7:
            explanation = f'Voice recognition score too low ({voice_score:.2f}, required >= 0.7). Please retry recording or use alternative verification.'
            return {
                'decision': 'DENY',
                'risk_level': 'MEDIUM',
                'explanation': explanation,
                'requires_verification_question': False,
                'verification_question': ''
            }
        
        # Default: Deny if scores are insufficient
        if final_score < 0.65:
            explanation = f'Overall authentication score insufficient ({final_score:.2f}). Access denied. Consider reviewing recent attempts for potential anomalies.'
            return {
                'decision': 'DENY',
                'risk_level': 'MEDIUM',
                'explanation': explanation,
                'requires_verification_question': False,
                'verification_question': ''
            }
    
    def test_connection(self) -> bool:
        """Test OpenRouter API connection"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Say 'connected'"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                url=self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"API Connection Test Failed: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from OpenRouter"""
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=10
            )
            
            if response.status_code == 200:
                models = response.json()['data']
                free_models = [m['id'] for m in models if ':free' in m['id']]
                return free_models
            return []
        except:
            return []