"""
Database Seeder Utility
Pre-seeds database with synthetic historical data for behavioral analysis
"""
import sqlite3
import json
import random
from datetime import datetime, timedelta
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def seed_database(num_records=100, clear_existing=False, output_file=None):
    """
    Seed database with synthetic historical data
    
    Args:
        num_records: Number of records to generate
        clear_existing: Whether to clear existing data first
        output_file: Optional JSON file to export the data
    """
    
    print(f"[SEED] Seeding database with {num_records} synthetic records...")
    
    # Connect to database
    conn = sqlite3.connect('smart_lock_events.db')
    cursor = conn.cursor()
    
    # Clear existing data if requested
    if clear_existing:
        cursor.execute("DELETE FROM access_events")
        print("🗑️  Cleared existing data")
    
    # Check how many records already exist
    cursor.execute("SELECT COUNT(*) FROM access_events")
    existing_count = cursor.fetchone()[0]
    
    if existing_count >= num_records and not clear_existing:
        print(f"[DATA] Database already has {existing_count} records")
        conn.close()
        return
    
    # Generate synthetic data
    base_date = datetime.now() - timedelta(days=30)
    synthetic_data = []
    
    for i in range(num_records):
        # Generate random date within last 30 days
        days_ago = random.randint(0, 29)
        event_date = base_date + timedelta(days=days_ago)
        
        # Generate time based on patterns
        # Normal hours: 70% during 7 AM - 10 PM
        if random.random() < 0.7:
            hour = random.choice([7, 8, 9, 10, 17, 18, 19, 20, 21])
        else:
            hour = random.randint(0, 23)
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        event_time = event_date.replace(hour=hour, minute=minute, second=second)
        
        # Determine if this is a normal or suspicious attempt
        is_normal = random.random() < 0.85  # 85% normal
        
        if is_normal:
            # Normal user patterns
            face_score = random.uniform(0.75, 0.98)
            voice_score = random.uniform(0.75, 0.98)
            pin_valid = True
            behavior_score = random.uniform(0.8, 1.0)
            risk_level = random.choice(['LOW', 'MEDIUM'])
            
            # 90% of normal attempts succeed
            if random.random() < 0.9:
                decision = 'ALLOW'
                explanation = "Normal access pattern with good biometric verification"
                failed_attempts = 0
            else:
                decision = 'DENY'
                explanation = random.choice([
                    "Slightly low biometric scores",
                    "Unusual time for access",
                    "Moderate risk detected"
                ])
                failed_attempts = random.randint(1, 2)
        else:
            # Suspicious patterns (15% of attempts)
            face_score = random.uniform(0.3, 0.7)
            voice_score = random.uniform(0.3, 0.7)
            pin_valid = random.random() < 0.7  # 70% chance correct PIN
            behavior_score = random.uniform(0.4, 0.7)
            risk_level = 'HIGH'
            
            # Most suspicious attempts fail
            if random.random() < 0.8:
                decision = 'DENY'
                explanation = random.choice([
                    "Multiple authentication failures",
                    "Biometric scores below threshold",
                    "Suspicious access pattern detected"
                ])
                failed_attempts = random.randint(2, 4)
            else:
                decision = 'ALLOW'  # False positive
                explanation = "Access granted despite some concerns"
                failed_attempts = 0
        
        # Calculate final score
        final_score = (
            0.3 * (1.0 if pin_valid else 0.0) +
            0.35 * face_score +
            0.35 * voice_score
        )
        
        # Determine lockout state
        lockout_active = failed_attempts >= 5
        
        # Prepare data for insertion
        data = {
            'action': random.choice(['LOCK', 'UNLOCK']),
            'entry_time': event_time.strftime('%Y-%m-%d %H:%M:%S') if random.random() > 0.5 else None,
            'exit_time': event_time.strftime('%Y-%m-%d %H:%M:%S') if random.random() > 0.5 else None,
            'pin_valid': pin_valid,
            'face_score': round(face_score, 3),
            'voice_score': round(voice_score, 3),
            'behavior_score': round(behavior_score, 3),
            'final_score': round(final_score, 3),
            'genai_decision': decision,
            'genai_risk_level': risk_level,
            'genai_explanation': explanation,
            'failed_attempt_count': failed_attempts,
            'lockout_active': lockout_active,
            'record_created_at': event_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        synthetic_data.append(data)
        
        # Insert into database
        cursor.execute('''
            INSERT INTO access_events (
                action, entry_time, exit_time, pin_valid,
                face_score, voice_score, behavior_score, final_score,
                genai_decision, genai_risk_level, genai_explanation,
                failed_attempt_count, lockout_active, record_created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['action'], data['entry_time'], data['exit_time'], data['pin_valid'],
            data['face_score'], data['voice_score'], data['behavior_score'], data['final_score'],
            data['genai_decision'], data['genai_risk_level'], data['genai_explanation'],
            data['failed_attempt_count'], data['lockout_active'], data['record_created_at']
        ))
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_records} records...")
    
    conn.commit()
    
    # Export to JSON if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(synthetic_data, f, indent=2, default=str)
        print(f"[FOLDER] Exported data to {output_file}")
    
    # Update statistics
    cursor.execute("SELECT COUNT(*) FROM access_events")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'ALLOW'")
    allowed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
    lockouts = cursor.fetchone()[0]
    
    print(f"\n[OK] Seeding complete!")
    print(f"[DATA] Total records: {total}")
    print(f"[OK] Allowed attempts: {allowed}")
    print(f"[ERROR] Lockout events: {lockouts}")
    print(f"[STATS] Success rate: {(allowed/total*100):.1f}%")
    
    conn.close()

def generate_advanced_patterns():
    """Generate more sophisticated behavioral patterns"""
    print("Generating advanced behavioral patterns...")
    
    conn = sqlite3.connect('smart_lock_events.db')
    cursor = conn.cursor()
    
    # Create specific patterns:
    
    # 1. Normal weekday morning pattern (7-9 AM)
    for day in range(5):  # Monday to Friday
        for hour in [7, 8, 9]:
            event_time = datetime.now() - timedelta(days=day+7, hours=random.randint(0, 23))
            event_time = event_time.replace(hour=hour, minute=random.randint(0, 59))
            
            cursor.execute('''
                INSERT INTO access_events (
                    action, entry_time, pin_valid,
                    face_score, voice_score, behavior_score, final_score,
                    genai_decision, genai_risk_level, genai_explanation,
                    failed_attempt_count, lockout_active, record_created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'UNLOCK',
                event_time.strftime('%Y-%m-%d %H:%M:%S'),
                True,
                0.92, 0.88, 0.95, 0.92,
                'ALLOW', 'LOW', 'Normal weekday morning access pattern',
                0, False, event_time.strftime('%Y-%m-%d %H:%M:%S')
            ))
    
    # 2. Weekend evening pattern (7-11 PM)
    for day in [5, 6]:  # Saturday, Sunday
        for hour in [19, 20, 21, 22]:
            event_time = datetime.now() - timedelta(days=day+7, hours=random.randint(0, 23))
            event_time = event_time.replace(hour=hour, minute=random.randint(0, 59))
            
            cursor.execute('''
                INSERT INTO access_events (
                    action, entry_time, pin_valid,
                    face_score, voice_score, behavior_score, final_score,
                    genai_decision, genai_risk_level, genai_explanation,
                    failed_attempt_count, lockout_active, record_created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'UNLOCK',
                event_time.strftime('%Y-%m-%d %H:%M:%S'),
                True,
                0.89, 0.91, 0.93, 0.90,
                'ALLOW', 'LOW', 'Weekend evening access pattern',
                0, False, event_time.strftime('%Y-%m-%d %H:%M:%S')
            ))
    
    # 3. Suspicious midnight attempts
    for i in range(5):
        event_time = datetime.now() - timedelta(days=random.randint(1, 30))
        event_time = event_time.replace(hour=random.choice([0, 1, 2, 3]), 
                                        minute=random.randint(0, 59))
        
        cursor.execute('''
            INSERT INTO access_events (
                action, entry_time, pin_valid,
                face_score, voice_score, behavior_score, final_score,
                genai_decision, genai_risk_level, genai_explanation,
                failed_attempt_count, lockout_active, record_created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            'UNLOCK',
            event_time.strftime('%Y-%m-%d %H:%M:%S'),
            random.random() < 0.5,
            0.45, 0.38, 0.35, 0.39,
            'DENY', 'HIGH', 'Suspicious midnight access attempt',
            random.randint(2, 4), False, event_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    conn.commit()
    conn.close()
    print("[OK] Advanced patterns generated!")

def analyze_database():
    """Analyze the current database contents"""
    print("\n[DATA] DATABASE ANALYSIS")
    print("="*50)
    
    conn = sqlite3.connect('smart_lock_events.db')
    cursor = conn.cursor()
    
    # Basic counts
    cursor.execute("SELECT COUNT(*) FROM access_events")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'ALLOW'")
    allowed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'DENY'")
    denied = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
    lockouts = cursor.fetchone()[0]
    
    # Time patterns
    cursor.execute("""
        SELECT strftime('%H', record_created_at) as hour, COUNT(*) as count
        FROM access_events 
        WHERE genai_decision = 'ALLOW'
        GROUP BY strftime('%H', record_created_at)
        ORDER BY count DESC
        LIMIT 5
    """)
    
    common_hours = cursor.fetchall()
    
    # Score statistics
    cursor.execute("""
        SELECT 
            AVG(face_score) as avg_face,
            MIN(face_score) as min_face,
            MAX(face_score) as max_face,
            AVG(voice_score) as avg_voice,
            MIN(voice_score) as min_voice,
            MAX(voice_score) as max_voice,
            AVG(final_score) as avg_final
        FROM access_events
    """)
    
    scores = cursor.fetchone()
    
    print(f"Total Records: {total}")
    print(f"Allowed: {allowed} ({allowed/total*100:.1f}%)")
    print(f"Denied: {denied} ({denied/total*100:.1f}%)")
    print(f"Lockouts: {lockouts}")
    print(f"\n[STATS] Score Statistics:")
    print(f"  Face: {scores[0]:.3f} avg ({scores[1]:.3f} min, {scores[2]:.3f} max)")
    print(f"  Voice: {scores[3]:.3f} avg ({scores[4]:.3f} min, {scores[5]:.3f} max)")
    print(f"  Final: {scores[6]:.3f} avg")
    
    print(f"\n[TIMER] Common Access Hours (Successful):")
    for hour, count in common_hours:
        print(f"  {hour}:00 - {count} attempts")
    
    # Check for patterns
    cursor.execute("""
        SELECT failed_attempt_count, COUNT(*) 
        FROM access_events 
        GROUP BY failed_attempt_count 
        ORDER BY failed_attempt_count
    """)
    
    failure_patterns = cursor.fetchall()
    
    print(f"\nFailure Patterns:")
    for attempts, count in failure_patterns:
        print(f"  {attempts} consecutive failures: {count} events")
    
    conn.close()
    
    return {
        'total': total,
        'allowed': allowed,
        'denied': denied,
        'lockouts': lockouts,
        'success_rate': allowed/total if total > 0 else 0
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seed database with synthetic data')
    parser.add_argument('--count', type=int, default=100, help='Number of records to generate')
    parser.add_argument('--clear', action='store_true', help='Clear existing data first')
    parser.add_argument('--export', type=str, help='Export data to JSON file')
    parser.add_argument('--advanced', action='store_true', help='Generate advanced patterns')
    parser.add_argument('--analyze', action='store_true', help='Analyze current database')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_database()
    else:
        seed_database(
            num_records=args.count,
            clear_existing=args.clear,
            output_file=args.export
        )
        
        if args.advanced:
            generate_advanced_patterns()