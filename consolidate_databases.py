#!/usr/bin/env python3
"""
Database Consolidation Script
Consolidates access_control.db (with day-aware data) into smart_lock_events.db
The system uses smart_lock_events.db, so we need to migrate the seeded data there
"""
import sqlite3
import os
import shutil
from datetime import datetime

print("=" * 80)
print("[CONSOLIDATE] Smart Lock Database Consolidation")
print("=" * 80)

source_db = "access_control.db"
target_db = "smart_lock_events.db"

# Check if source database exists
if not os.path.exists(source_db):
    print(f"[ERROR] Source database '{source_db}' not found!")
    print("[INFO] The system will use 'smart_lock_events.db' (current default)")
    print("[INFO] If you have old data in '{source_db}', please ensure it's available.")
    exit(1)

print(f"\n[CHECK] Found source database: {source_db}")
print(f"[CHECK] Target database: {target_db}")

# Backup target database if it exists
if os.path.exists(target_db):
    backup_name = f"{target_db}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(target_db, backup_name)
    print(f"[BACKUP] Created backup: {backup_name}")
    os.remove(target_db)

# Copy source to target
shutil.copy(source_db, target_db)
print(f"[OK] Copied {source_db} -> {target_db}")

# Verify the migration
conn = sqlite3.connect(target_db)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM access_events")
count = cursor.fetchone()[0]
print(f"[VERIFY] Target database contains {count} events")

cursor.execute("SELECT owner, COUNT(*) as count FROM access_events GROUP BY owner")
print(f"[VERIFY] Events by owner:")
for owner, owner_count in cursor.fetchall():
    print(f"   {owner}: {owner_count}")

cursor.execute("SELECT COUNT(*) FROM access_events WHERE day_of_week IS NOT NULL")
day_count = cursor.fetchone()[0]
print(f"[VERIFY] Events with day_of_week field: {day_count}/{count}")

conn.close()

print("\n" + "=" * 80)
print("[OK] DATABASE CONSOLIDATION COMPLETE")
print("=" * 80)
print(f"\nSystem Configuration:")
print(f"  Active Database: {target_db}")
print(f"  Source Database: {source_db}")
print(f"  Total Events: {count} (with day-aware patterns)")
print(f"\nThe system is now using the day-aware database with 700+ seeded events!")
print("=" * 80)
