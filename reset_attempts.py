"""
Reset system to clean state - removes new login attempts but keeps original database
"""
import sqlite3
import os

def reset_system():
    db_path = 'smart_lock_events.db'
    
    if not os.path.exists(db_path):
        print("[ERROR] Database not found")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First, check how many original events there are
        cursor.execute("SELECT COUNT(*) FROM access_events")
        total_count = cursor.fetchone()[0]
        print(f"[DATA] Current total events: {total_count}")
        
        # The original seeded database had 99 events
        # Delete all events beyond the original 99
        if total_count > 99:
            cursor.execute("DELETE FROM access_events WHERE rowid > 99")
            deleted = cursor.rowcount
            print(f"🗑️  Deleted {deleted} new login attempts")
        else:
            print("[OK] No additional events to delete")
        
        # Reset all lockout states
        cursor.execute("UPDATE access_events SET lockout_active = 0")
        cursor.execute("UPDATE access_events SET failed_attempt_count = 0")
        
        conn.commit()
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM access_events")
        final_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
        lockout_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"[OK] Final event count: {final_count}")
        print(f"[OK] Active lockouts: {lockout_count}")
        print(f"[OK] System reset complete!")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reset_system()
