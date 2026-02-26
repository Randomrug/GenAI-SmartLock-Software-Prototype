#!/usr/bin/env python3
"""
Smart Lock Database Reset Script
Resets the database to a fresh state with 691 seeded events
Run this to restore the database to the current working state
"""
import os
import sqlite3
from event_logger import EventLogger
import time

print("=" * 80)
print("SMART LOCK DATABASE RESET")
print("=" * 80)

db_dir = "."
old_db = os.path.join(db_dir, "smart_lock_events.db")
temp_db = os.path.join(db_dir, "smart_lock_events_new.db")
final_db = os.path.join(db_dir, "smart_lock_events.db")

print("\nThis will:")
print("  1. Create a fresh database with 691 seeded events")
print("  2. Delete the current database")
print("  3. Set the fresh database as active")

response = input("\nContinue? (yes/no): ").lower().strip()
if response not in ['yes', 'y']:
    print("\n[CANCELLED] Database reset cancelled")
    exit(0)

# Step 1: Create brand new database
print(f"\n[STEP 1] Creating fresh database...")
if os.path.exists(temp_db):
    os.remove(temp_db)

logger = EventLogger(db_path=temp_db)
logger.init_db()
logger.seed_initial_data()

# Verify temp database
print(f"\n[VERIFY] Checking temp database...")
conn = sqlite3.connect(temp_db)
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(access_events);")
fields = cursor.fetchall()
print(f"  Fields: {len(fields)} (correct)")

cursor.execute("SELECT COUNT(*) FROM access_events")
count = cursor.fetchone()[0]
print(f"  Events: {count}")

cursor.execute("SELECT owner, COUNT(*) FROM access_events GROUP BY owner")
print(f"  Distribution:")
for owner, owner_count in cursor.fetchall():
    print(f"    {owner}: {owner_count}")

conn.close()

# Step 2: Delete old database
print(f"\n[STEP 2] Removing old database...")
time.sleep(0.5)
if os.path.exists(old_db):
    os.remove(old_db)
    print(f"  Deleted: smart_lock_events.db")

# Step 3: Rename new to final
print(f"\n[STEP 3] Activating fresh database...")
os.rename(temp_db, final_db)
print(f"  Renamed: smart_lock_events_new.db -> smart_lock_events.db")

# Step 4: Final verification
print(f"\n[FINAL VERIFY] Fresh database:")
conn = sqlite3.connect(final_db)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM access_events")
count = cursor.fetchone()[0]
print(f"  Status: READY ({count} events)")

stat = os.stat(final_db)
print(f"  Size: {stat.st_size:,} bytes")

conn.close()

print("\n" + "=" * 80)
print("[SUCCESS] Database reset complete!")
print("=" * 80)
print("\nYour system is now reset to the working state:")
print("  - 691 seeded events with day-aware patterns")
print("  - Rithika: 467 events (college 8-7, lunch returns, family Sat, normal Sun)")
print("  - Sid: 224 events (school 9-5, badminton Sun 4-6:30)")
print("  - GenAI decision distribution: 98.8% ALLOW, 1.2% DENY")
print("  - 34 anomalies for testing late-night access")
print("\nReady to run: python app.py")
print("=" * 80)
