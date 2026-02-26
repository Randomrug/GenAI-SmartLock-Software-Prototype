#!/usr/bin/env python3
"""Reset lockout state"""
import sqlite3

conn = sqlite3.connect('smart_lock_events.db')
cursor = conn.cursor()

# Clear lockout
cursor.execute('UPDATE access_events SET lockout_active = 0')
cursor.execute('UPDATE access_events SET failed_attempt_count = 0')

conn.commit()

cursor.execute('SELECT COUNT(*) FROM access_events WHERE lockout_active = 1')
lockout_count = cursor.fetchone()[0]

conn.close()

print(f'✓ Lockout cleared')
print(f'✓ Remaining lockout records: {lockout_count}')
print(f'✓ SYSTEM UNLOCKED - Refresh browser to see changes')
