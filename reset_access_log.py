#!/usr/bin/env python3
"""
Reset access log while keeping PIN and configuration
"""
import sqlite3
import sys

def reset_access_log():
    """Clear all access event logs"""
    try:
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        
        # Get count before
        cursor.execute("SELECT COUNT(*) FROM access_events")
        before = cursor.fetchone()[0]
        print(f"[DATA] Current logs: {before} events")
        
        # Clear all events
        cursor.execute("DELETE FROM access_events")
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM access_events")
        after = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"[OK] Access log cleared: {before} → {after} events")
        return True
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

def reset_lockout_only():
    """Just clear lockout state without deleting logs"""
    try:
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        
        # Clear lockout flags
        cursor.execute('UPDATE access_events SET lockout_active = 0')
        cursor.execute('UPDATE access_events SET failed_attempt_count = 0')
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
        remaining = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"[OK] Lockout cleared / Remaining locked records: {remaining}")
        return True
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

if __name__ == "__main__":
    print("\n[RETRY] Access Log Reset Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "clear":
            reset_access_log()
        elif sys.argv[1] == "lockout":
            reset_lockout_only()
    else:
        print("\nOptions:")
        print("  python reset_access_log.py clear    → Delete ALL access logs")
        print("  python reset_access_log.py lockout  → Clear just lockout state")
        print("\nUsage: python reset_access_log.py <option>")
