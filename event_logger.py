"""
SQLite Database Interface for Smart Lock Events
Persistent storage with pre-seeded historical data
"""
import sqlite3
import json
import os
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional, Any

class EventLogger:
    def __init__(self, db_path: str = 'smart_lock_events.db'):
        self.db_path = db_path
        self.init_db()
        self.seed_initial_data()
    
    def init_db(self) -> None:
        """Initialize database with required tables - IN/OUT access tracking"""
        print("[CHART] Initializing database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop existing table to start fresh
        cursor.execute('DROP TABLE IF EXISTS access_events')
        
        # Create main access events table with IN/OUT actions and new columns for advanced pipeline
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner TEXT CHECK(owner IN ('rithika', 'sid', 'unknown')),
                action TEXT NOT NULL CHECK(action IN ('IN', 'OUT')),
                day_of_week TEXT CHECK(day_of_week IN ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')),
                entry_time TEXT,
                exit_time TEXT,
                event_datetime TEXT, -- NEW: stores manual_datetime
                person_status TEXT CHECK(person_status IN ('INSIDE', 'OUTSIDE')),
                pin_valid BOOLEAN NOT NULL,
                face_score REAL CHECK(face_score >= 0 AND face_score <= 1),
                voice_score REAL CHECK(face_score >= 0 AND face_score <= 1),
                face_owner TEXT,
                voice_owner TEXT,
                behavior_score REAL CHECK(behavior_score >= 0 AND behavior_score <= 1),
                final_score REAL CHECK(final_score >= 0 AND final_score <= 1),
                genai_decision TEXT NOT NULL CHECK(genai_decision IN ('ALLOW', 'DENY', 'LOCKOUT')),
                genai_risk_level TEXT NOT NULL CHECK(genai_risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
                genai_explanation TEXT NOT NULL,
                failed_attempt_count INTEGER DEFAULT 0,
                lockout_active BOOLEAN DEFAULT 0,
                model_explanation TEXT,
                owner_feedback TEXT,
                genai_updated_explanation TEXT,
                alert_sent BOOLEAN DEFAULT 0,
                alert_type TEXT,
                door_status TEXT,
                record_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_decision 
            ON access_events(genai_decision, record_created_at)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_owner 
            ON access_events(owner, record_created_at)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_scores 
            ON access_events(face_score, voice_score, final_score)
        ''')
        
        # Create statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_name TEXT UNIQUE NOT NULL,
                stat_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print("[OK] Database initialized successfully with IN/OUT tracking")
    
    def seed_initial_data(self) -> None:
        """Pre-seed database with 200+ synthetic events with day-specific owner patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we already have data
            cursor.execute("SELECT COUNT(*) FROM access_events")
            count = cursor.fetchone()[0]
            
            if count == 0:
                print("[SEED] Seeding database with 200+ day-aware owner patterns...")
                
                # Day-specific owner schedule definitions
                owner_schedules = {
                    'rithika': {
                        'Monday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'description': 'College 8-7 (lunch 1-2)'},
                        'Tuesday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'description': 'College 8-7 (lunch 1-2)'},
                        'Wednesday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'description': 'College 8-7 (lunch 1-2)'},
                        'Thursday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'description': 'College 8-7 (lunch 1-2)'},
                        'Friday': {'start': 8, 'end': 19, 'lunch': (13, 14), 'description': 'College 8-7 (lunch 1-2)'},
                        'Saturday': {'start': 10, 'end': 18, 'lunch': None, 'description': 'Family events'},
                        'Sunday': {'start': 9, 'end': 21, 'lunch': None, 'description': 'Normal home day'}
                    },
                    'sid': {
                        'Monday': {'start': 9, 'end': 17, 'lunch': None, 'description': 'School 9-5'},
                        'Tuesday': {'start': 9, 'end': 17, 'lunch': None, 'description': 'School 9-5'},
                        'Wednesday': {'start': 9, 'end': 17, 'lunch': None, 'description': 'School 9-5'},
                        'Thursday': {'start': 9, 'end': 17, 'lunch': None, 'description': 'School 9-5'},
                        'Friday': {'start': 9, 'end': 17, 'lunch': None, 'description': 'School 9-5'},
                        'Saturday': {'start': 10, 'end': 20, 'lunch': None, 'description': 'Weekend activities'},
                        'Sunday': {'start': 8, 'end': 22, 'lunch': None, 'description': 'Badminton 4-6:30 PM'}
                    }
                }
                
                # Generate 200+ entries
                base_date = datetime.now() - timedelta(days=60)
                events = []
                
                # Generate events for each day across 4 weeks
                for days_offset in range(60):
                    event_date = base_date + timedelta(days=days_offset)
                    day_name = event_date.strftime('%A')
                    
                    # Distribute events: 60% Rithika, 40% Sid
                    for owner in ['rithika', 'rithika', 'rithika', 'sid', 'sid']:
                        schedule = owner_schedules[owner][day_name]
                        
                        # Morning entrance (8-10 AM for Rithika weekdays, 9-10 for Sid)
                        if random.random() < 0.85:  # 85% probability of morning entrance
                            entrance_hour = random.randint(schedule['start'], min(schedule['start'] + 2, 10))
                            entrance_minute = random.randint(0, 59)
                            entrance_time = event_date.replace(hour=entrance_hour, minute=entrance_minute)
                            
                            # Good biometric scores for normal entrance
                            face_score = random.uniform(0.80, 0.95)
                            voice_score = random.uniform(0.82, 0.95)
                            pin_valid = True
                            behavior_score = 0.9
                            
                            final_score = 0.3 * (1.0 if pin_valid else 0.0) + 0.35 * face_score + 0.35 * voice_score
                            
                            events.append({
                                'owner': owner,
                                'action': 'IN',
                                'day_of_week': day_name,
                                'entry_time': entrance_time,
                                'exit_time': None,
                                'person_status': 'INSIDE',
                                'pin_valid': pin_valid,
                                'face_score': face_score,
                                'voice_score': voice_score,
                                'face_owner': owner,
                                'voice_owner': owner,
                                'behavior_score': behavior_score,
                                'final_score': final_score,
                                'genai_decision': 'ALLOW',
                                'genai_risk_level': 'LOW',
                                'genai_explanation': f"Normal morning entrance for {owner} on {day_name}",
                                'failed_attempt_count': 0,
                                'lockout_active': False,
                                'record_created_at': entrance_time
                            })
                        
                        # Lunch return for Rithika on weekdays (50% chance)
                        if owner == 'rithika' and day_name in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                            if random.random() < 0.5:
                                lunch_time = event_date.replace(hour=13, minute=random.randint(5, 15))
                                face_score = random.uniform(0.85, 0.95)
                                voice_score = random.uniform(0.85, 0.95)
                                pin_valid = True
                                behavior_score = 0.92
                                final_score = 0.3 * (1.0) + 0.35 * face_score + 0.35 * voice_score
                                
                                events.append({
                                    'owner': owner,
                                    'action': 'IN',
                                        'day_of_week': day_name,
                                    'entry_time': lunch_time,
                                    'exit_time': None,
                                    'person_status': 'INSIDE',
                                    'pin_valid': pin_valid,
                                    'face_score': face_score,
                                    'voice_score': voice_score,
                                    'face_owner': owner,
                                    'voice_owner': owner,
                                    'behavior_score': behavior_score,
                                    'final_score': final_score,
                                    'genai_decision': 'ALLOW',
                                    'genai_risk_level': 'LOW',
                                    'genai_explanation': f"Lunch break return for {owner}. Expected 1-2 PM on weekdays.",
                                    'failed_attempt_count': 0,
                                    'lockout_active': False,
                                    'record_created_at': lunch_time
                                })
                                
                                # Lunch exit (back to college)
                                exit_time = event_date.replace(hour=14, minute=random.randint(0, 30))
                                face_score = random.uniform(0.85, 0.95)
                                voice_score = random.uniform(0.85, 0.95)
                                
                                events.append({
                                    'owner': owner,
                                    'action': 'OUT',
                                        'day_of_week': day_name,
                                    'entry_time': None,
                                    'exit_time': exit_time,
                                    'person_status': 'OUTSIDE',
                                    'pin_valid': pin_valid,
                                    'face_score': face_score,
                                    'voice_score': voice_score,
                                    'face_owner': owner,
                                    'voice_owner': owner,
                                    'behavior_score': behavior_score,
                                    'final_score': final_score,
                                    'genai_decision': 'ALLOW',
                                    'genai_risk_level': 'LOW',
                                    'genai_explanation': f"Exiting for college. Expected pattern for {owner}.",
                                    'failed_attempt_count': 0,
                                    'lockout_active': False,
                                    'record_created_at': exit_time
                                })
                        
                        # Evening exit
                        if random.random() < 0.90:  # 90% probability of evening exit
                            exit_hour = random.randint(schedule['end'] - 2, schedule['end'])
                            exit_minute = random.randint(0, 59)
                            exit_time = event_date.replace(hour=exit_hour, minute=exit_minute)
                            
                            face_score = random.uniform(0.80, 0.95)
                            voice_score = random.uniform(0.82, 0.95)
                            pin_valid = True
                            behavior_score = 0.9
                            final_score = 0.3 * (1.0) + 0.35 * face_score + 0.35 * voice_score
                            
                            events.append({
                                'owner': owner,
                                'action': 'OUT',
                                'day_of_week': day_name,
                                'entry_time': None,
                                'exit_time': exit_time,
                                'person_status': 'OUTSIDE',
                                'pin_valid': pin_valid,
                                'face_score': face_score,
                                'voice_score': voice_score,
                                'face_owner': owner,
                                'voice_owner': owner,
                                'behavior_score': behavior_score,
                                'final_score': final_score,
                                'genai_decision': 'ALLOW',
                                'genai_risk_level': 'LOW',
                                'genai_explanation': f"Evening exit for {owner} on {day_name}. Normal pattern.",
                                'failed_attempt_count': 0,
                                'lockout_active': False,
                                'record_created_at': exit_time
                            })
                        
                        # Sid's Sunday badminton (4-6:30 PM)
                        if owner == 'sid' and day_name == 'Sunday':
                            if random.random() < 0.7:
                                badminton_exit = event_date.replace(hour=16, minute=random.randint(0, 30))
                                face_score = random.uniform(0.85, 0.95)
                                voice_score = random.uniform(0.85, 0.95)
                                pin_valid = True
                                behavior_score = 0.91
                                final_score = 0.3 * (1.0) + 0.35 * face_score + 0.35 * voice_score
                                
                                events.append({
                                    'owner': owner,
                                    'action': 'OUT',
                                        'day_of_week': day_name,
                                    'entry_time': None,
                                    'exit_time': badminton_exit,
                                    'person_status': 'OUTSIDE',
                                    'pin_valid': pin_valid,
                                    'face_score': face_score,
                                    'voice_score': voice_score,
                                    'face_owner': owner,
                                    'voice_owner': owner,
                                    'behavior_score': behavior_score,
                                    'final_score': final_score,
                                    'genai_decision': 'ALLOW',
                                    'genai_risk_level': 'LOW',
                                    'genai_explanation': f"Sunday badminton class. Expected 4-6:30 PM on Sundays.",
                                    'failed_attempt_count': 0,
                                    'lockout_active': False,
                                    'record_created_at': badminton_exit
                                })
                                
                                # Badminton return (6-7 PM)
                                badminton_return = event_date.replace(hour=random.randint(18, 19), minute=random.randint(0, 59))
                                
                                events.append({
                                    'owner': owner,
                                    'action': 'IN',
                                        'day_of_week': day_name,
                                    'entry_time': badminton_return,
                                    'exit_time': None,
                                    'person_status': 'INSIDE',
                                    'pin_valid': pin_valid,
                                    'face_score': face_score,
                                    'voice_score': voice_score,
                                    'face_owner': owner,
                                    'voice_owner': owner,
                                    'behavior_score': behavior_score,
                                    'final_score': final_score,
                                    'genai_decision': 'ALLOW',
                                    'genai_risk_level': 'LOW',
                                    'genai_explanation': f"Return from badminton class. Normal Sunday pattern for {owner}.",
                                    'failed_attempt_count': 0,
                                    'lockout_active': False,
                                    'record_created_at': badminton_return
                                })
                        
                        # Rithika's Saturday family events
                        if owner == 'rithika' and day_name == 'Saturday':
                            if random.random() < 0.6:
                                family_exit = event_date.replace(hour=random.randint(10, 12), minute=random.randint(0, 59))
                                face_score = random.uniform(0.85, 0.95)
                                voice_score = random.uniform(0.85, 0.95)
                                pin_valid = True
                                behavior_score = 0.91
                                final_score = 0.3 * (1.0) + 0.35 * face_score + 0.35 * voice_score
                                
                                events.append({
                                    'owner': owner,
                                    'action': 'OUT',
                                        'day_of_week': day_name,
                                    'entry_time': None,
                                    'exit_time': family_exit,
                                    'person_status': 'OUTSIDE',
                                    'pin_valid': pin_valid,
                                    'face_score': face_score,
                                    'voice_score': voice_score,
                                    'face_owner': owner,
                                    'voice_owner': owner,
                                    'behavior_score': behavior_score,
                                    'final_score': final_score,
                                    'genai_decision': 'ALLOW',
                                    'genai_risk_level': 'LOW',
                                    'genai_explanation': f"Saturday family event. Normal weekend pattern for {owner}.",
                                    'failed_attempt_count': 0,
                                    'lockout_active': False,
                                    'record_created_at': family_exit
                                })
                                
                                # Family return (evening)
                                family_return = event_date.replace(hour=random.randint(17, 19), minute=random.randint(0, 59))
                                
                                events.append({
                                    'owner': owner,
                                    'action': 'IN',
                                        'day_of_week': day_name,
                                    'entry_time': family_return,
                                    'exit_time': None,
                                    'person_status': 'INSIDE',
                                    'pin_valid': pin_valid,
                                    'face_score': face_score,
                                    'voice_score': voice_score,
                                    'face_owner': owner,
                                    'voice_owner': owner,
                                    'behavior_score': behavior_score,
                                    'final_score': final_score,
                                    'genai_decision': 'ALLOW',
                                    'genai_risk_level': 'LOW',
                                    'genai_explanation': f"Return from family event. Normal Saturday pattern for {owner}.",
                                    'failed_attempt_count': 0,
                                    'lockout_active': False,
                                    'record_created_at': family_return
                                })
                
                # Add some anomalous entries for testing (5% of total)
                anomaly_count = len(events) // 20  # 5% anomalies
                for _ in range(anomaly_count):
                    event = random.choice(events)
                    owner = event['owner']
                    day_name = event['day_of_week']
                    
                    # Create anomaly: late night access
                    if random.random() < 0.5:
                        anomaly_time = event['record_created_at'].replace(hour=random.randint(23, 23), minute=random.randint(30, 59))
                        face_score = random.uniform(0.65, 0.75)
                        voice_score = random.uniform(0.60, 0.75)
                        pin_valid = random.random() < 0.7
                        behavior_score = random.uniform(0.5, 0.7)
                        final_score = 0.3 * (1.0 if pin_valid else 0.0) + 0.35 * face_score + 0.35 * voice_score
                        
                        events.append({
                            'owner': owner,
                            'action': 'IN',
                            'location': 'MAIN_DOOR',
                            'day_of_week': day_name,
                            'entry_time': anomaly_time,
                            'exit_time': None,
                            'person_status': 'INSIDE',
                            'pin_valid': pin_valid,
                            'face_score': face_score,
                            'voice_score': voice_score,
                            'face_owner': owner,
                            'voice_owner': owner,
                            'behavior_score': behavior_score,
                            'final_score': final_score,
                            'genai_decision': 'DENY' if final_score < 0.7 else 'ALLOW',
                            'genai_risk_level': 'HIGH',
                            'genai_explanation': f"Late night access attempt for {owner} at {anomaly_time.strftime('%H:%M')}. Anomalous time outside typical schedule.",
                            'failed_attempt_count': 1 if final_score < 0.7 else 0,
                            'lockout_active': False,
                            'record_created_at': anomaly_time
                        })
                
                # Sort events by timestamp and insert
                events.sort(key=lambda x: x['record_created_at'])
                
                for event in events:
                    cursor.execute('''
                        INSERT INTO access_events (
                            owner, action, day_of_week, entry_time, exit_time, 
                            person_status, pin_valid, face_score, voice_score, face_owner, voice_owner,
                            behavior_score, final_score, genai_decision, genai_risk_level, 
                            genai_explanation, failed_attempt_count, lockout_active, record_created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event['owner'], event['action'], event['day_of_week'],
                        event['entry_time'].strftime('%Y-%m-%d %H:%M:%S') if event['entry_time'] else None,
                        event['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if event['exit_time'] else None,
                        event['person_status'], event['pin_valid'], event['face_score'], event['voice_score'],
                        event['face_owner'], event['voice_owner'], event['behavior_score'], event['final_score'],
                        event['genai_decision'], event['genai_risk_level'], event['genai_explanation'],
                        event['failed_attempt_count'], event['lockout_active'],
                        event['record_created_at'].strftime('%Y-%m-%d %H:%M:%S')
                    ))
                
                conn.commit()
                print(f"[OK] Seeded {len(events)} historical records with day-specific patterns")
                print(f"   • Rithika: Mon-Fri 8-7 (lunch 1-2), Sat family events, Sun normal")
                print(f"   • Sid: Mon-Fri 9-5, Sun badminton 4-6:30 PM")
                print(f"   • Anomalies: {anomaly_count} late-night access attempts for testing")
                
                # Update statistics
                self._update_statistics(cursor)
                
            else:
                print(f"[CHART] Database already contains {count} records")
            
            conn.close()
            
        except Exception as e:
            print(f"[ERROR] Error seeding database: {e}")
    
    def _update_statistics(self, cursor) -> None:
        """Update system statistics"""
        try:
            # Calculate various statistics
            cursor.execute("SELECT COUNT(*) FROM access_events")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'ALLOW'")
            allowed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'DENY'")
            denied = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
            lockouts = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(face_score) FROM access_events WHERE genai_decision = 'ALLOW'")
            avg_face = cursor.fetchone()[0] or 0.0
            
            cursor.execute("SELECT AVG(voice_score) FROM access_events WHERE genai_decision = 'ALLOW'")
            avg_voice = cursor.fetchone()[0] or 0.0
            
            stats = {
                'total_events': total,
                'allowed_events': allowed,
                'denied_events': denied,
                'lockout_events': lockouts,
                'success_rate': allowed / total if total > 0 else 0,
                'avg_face_score_success': avg_face,
                'avg_voice_score_success': avg_voice
            }
            
            # Insert/update statistics
            for stat_name, stat_value in stats.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO system_statistics (stat_name, stat_value)
                    VALUES (?, ?)
                ''', (stat_name, str(stat_value)))
            
        except Exception as e:
            print(f"Warning: Could not update statistics: {e}")
    
    def log_event(self, 
                  action: str,
                  day_of_week: str = None,
                  entry_time: Optional[str] = None,
                  exit_time: Optional[str] = None,
                  event_datetime: Optional[str] = None, # NEW
                  pin_valid: bool = False,
                  face_score: float = 0.0,
                  voice_score: float = 0.0,
                  behavior_score: float = 0.0,
                  final_score: float = 0.0,
                  genai_decision: str = 'DENY',
                  genai_risk_level: str = 'MEDIUM',
                  genai_explanation: str = '',
                  failed_attempt_count: int = 0,
                  lockout_active: bool = False,
                  owner: str = 'unknown',
                  person_status: str = 'OUTSIDE',
                  face_owner: str = 'unknown',
                  voice_owner: str = 'unknown',
                  door_status: str = 'UNKNOWN',
                  model_explanation: str = None,
                  owner_feedback: str = None,
                  genai_updated_explanation: str = None,
                  alert_sent: bool = False,
                  alert_type: str = None) -> int:
        """
        Log an access attempt to the database
        
        Args:
            owner: Owner attempting access (rithika, sid, unknown)
            door_status: Current door status (LOCKED, UNLOCKED)
            face_owner: Detected owner from face verification
            voice_owner: Detected owner from voice verification
        
        Returns:
            event_id: ID of the created record
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # SQLite cannot bind dict/list directly. Normalize rich fields to JSON text.
            model_explanation_db = (
                json.dumps(model_explanation, ensure_ascii=False)
                if isinstance(model_explanation, (dict, list))
                else model_explanation
            )
            owner_feedback_db = (
                json.dumps(owner_feedback, ensure_ascii=False)
                if isinstance(owner_feedback, (dict, list))
                else owner_feedback
            )
            genai_updated_explanation_db = (
                json.dumps(genai_updated_explanation, ensure_ascii=False)
                if isinstance(genai_updated_explanation, (dict, list))
                else genai_updated_explanation
            )
            
            cursor.execute('''
                INSERT INTO access_events (
                    owner, action, day_of_week, entry_time, exit_time, event_datetime, person_status, pin_valid,
                    face_score, voice_score, face_owner, voice_owner,
                    behavior_score, final_score,
                    genai_decision, genai_risk_level, genai_explanation,
                    failed_attempt_count, lockout_active,
                    model_explanation, owner_feedback, genai_updated_explanation,
                    alert_sent, alert_type, door_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                owner, action, day_of_week, entry_time, exit_time, event_datetime, person_status, pin_valid,
                face_score, voice_score, face_owner, voice_owner,
                behavior_score, final_score,
                genai_decision, genai_risk_level, genai_explanation,
                failed_attempt_count, lockout_active,
                model_explanation_db, owner_feedback_db, genai_updated_explanation_db,
                alert_sent, alert_type, door_status
            ))
            
            event_id = cursor.lastrowid
            
            # Update statistics
            self._update_statistics(cursor)
            
            conn.commit()
            conn.close()
            
            # Log to text file for backup
            self._log_to_file({
                'timestamp': datetime.now().isoformat(),
                'event_id': event_id,
                'owner': owner,
                'face_owner': face_owner,
                'voice_owner': voice_owner,
                'action': action,
                'door_status': door_status,
                'decision': genai_decision,
                'risk_level': genai_risk_level,
                'explanation': genai_explanation,
                'face_score': face_score,
                'voice_score': voice_score,
                'final_score': final_score
            })
            
            return event_id
            
        except Exception as e:
            print(f"[ERROR] Error logging event: {e}")
            return -1
    
    def _log_to_file(self, event_data: Dict) -> None:
        """Log event to text file for backup"""
        try:
            os.makedirs('logs', exist_ok=True)
            
            log_file = 'logs/attempts.log'
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"Warning: Could not write to log file: {e}")
    
    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get recent access events"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM access_events 
                ORDER BY record_created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Error getting recent events: {e}")
            return []
    
    def get_failure_streak(self) -> int:
        """Get current consecutive failure count"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get the most recent failed attempt count
            cursor.execute('''
                SELECT failed_attempt_count FROM access_events 
                ORDER BY record_created_at DESC LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 0
        except:
            return 0
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get from statistics table
            cursor.execute("SELECT stat_name, stat_value FROM system_statistics")
            rows = cursor.fetchall()
            
            stats = {}
            for name, value in rows:
                try:
                    # Try to convert numeric values
                    if '.' in value:
                        stats[name] = float(value)
                    else:
                        stats[name] = int(value)
                except:
                    stats[name] = value
            
            # If statistics table is empty, calculate fresh
            if not stats:
                stats = self._calculate_fresh_statistics(cursor)
            
            conn.close()
            
            # Add current time
            stats['last_updated'] = datetime.now().isoformat()
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def _calculate_fresh_statistics(self, cursor) -> Dict:
        """Calculate statistics from scratch"""
        cursor.execute("SELECT COUNT(*) FROM access_events")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'ALLOW'")
        allowed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'DENY'")
        denied = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
        lockouts = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(face_score) FROM access_events WHERE genai_decision = 'ALLOW'")
        avg_face = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(voice_score) FROM access_events WHERE genai_decision = 'ALLOW'")
        avg_voice = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(final_score) FROM access_events WHERE genai_decision = 'ALLOW'")
        avg_final = cursor.fetchone()[0] or 0.0
        
        return {
            'total_events': total,
            'allowed_events': allowed,
            'denied_events': denied,
            'lockout_events': lockouts,
            'success_rate': allowed / total if total > 0 else 0,
            'avg_face_score_success': avg_face,
            'avg_voice_score_success': avg_voice,
            'avg_final_score_success': avg_final
        }
    
    def reset_lockout(self) -> bool:
        """Reset lockout state"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE access_events 
                SET lockout_active = 0 
                WHERE lockout_active = 1
            """)
            
            rows_affected = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if rows_affected > 0:
                print(f"[OK] Reset lockout on {rows_affected} records")
                return True
            else:
                print("[WARNING]  No active lockout found to reset")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error resetting lockout: {e}")
            return False
    
    def get_behavior_patterns(self) -> Dict:
        """Extract behavioral patterns for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get time patterns for successful unlocks
            cursor.execute('''
                SELECT 
                    strftime('%H', record_created_at) as hour,
                    COUNT(*) as count
                FROM access_events 
                WHERE action = 'IN' 
                  AND genai_decision = 'ALLOW'
                GROUP BY strftime('%H', record_created_at)
                ORDER BY count DESC
            ''')
            
            time_patterns = cursor.fetchall()
            
            # Get average session duration
            cursor.execute('''
                SELECT 
                    AVG(
                        CAST(
                            (strftime('%s', exit_time) - strftime('%s', entry_time)) / 3600.0 
                            AS REAL
                        )
                    ) as avg_hours
                FROM access_events 
                WHERE entry_time IS NOT NULL 
                  AND exit_time IS NOT NULL
                  AND exit_time > entry_time
            ''')
            
            avg_duration = cursor.fetchone()[0] or 1.0
            
            # Get score distributions
            cursor.execute('''
                SELECT 
                    AVG(face_score) as avg_face,
                    STDDEV(face_score) as std_face,
                    AVG(voice_score) as avg_voice,
                    STDDEV(voice_score) as std_voice,
                    AVG(final_score) as avg_final,
                    STDDEV(final_score) as std_final
                FROM access_events 
                WHERE genai_decision = 'ALLOW'
            ''')
            
            score_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                'common_access_hours': [int(hour) for hour, count in time_patterns[:5]],
                'avg_session_duration_hours': float(avg_duration) if avg_duration else 1.0,
                'score_statistics': {
                    'face_mean': score_stats[0] if score_stats[0] else 0.8,
                    'face_std': score_stats[1] if score_stats[1] else 0.1,
                    'voice_mean': score_stats[2] if score_stats[2] else 0.8,
                    'voice_std': score_stats[3] if score_stats[3] else 0.1,
                    'final_mean': score_stats[4] if score_stats[4] else 0.8,
                    'final_std': score_stats[5] if score_stats[5] else 0.1
                }
            }
            
        except Exception as e:
            print(f"Error getting behavior patterns: {e}")
            return {}
    
    def export_to_csv(self, filename: str = 'export.csv') -> bool:
        """Export database to CSV file"""
        try:
            import csv
            
            events = self.get_recent_events(limit=1000)  # Export up to 1000 records
            
            if not events:
                print("No data to export")
                return False
            
            # Define CSV headers
            headers = [
                'id', 'action', 'entry_time', 'exit_time', 'pin_valid',
                'face_score', 'voice_score', 'behavior_score', 'final_score',
                'genai_decision', 'genai_risk_level', 'genai_explanation',
                'failed_attempt_count', 'lockout_active', 'record_created_at'
            ]
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for event in events:
                    # Convert boolean values
                    row = {k: event.get(k) for k in headers}
                    row['pin_valid'] = 1 if row['pin_valid'] else 0
                    row['lockout_active'] = 1 if row['lockout_active'] else 0
                    writer.writerow(row)
            
            print(f"[OK] Exported {len(events)} records to {filename}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error exporting to CSV: {e}")
            return False
