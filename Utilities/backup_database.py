"""
Database Backup Utility
Automated database backup and restoration
"""
import sqlite3
import os
import shutil
import json
import csv
from datetime import datetime, timedelta
import argparse
import sys
import zipfile

class DatabaseBackup:
    def __init__(self, db_path='smart_lock_events.db', backup_dir='backups'):
        self.db_path = db_path
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
    
    def create_backup(self, compress=True, include_logs=True):
        """
        Create a comprehensive backup of the database
        
        Args:
            compress: Whether to compress the backup
            include_logs: Whether to include log files
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"smart_lock_backup_{timestamp}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            print(f"[BACKUP] Creating backup: {backup_name}")
            
            # 1. Backup the database file
            db_backup_path = f"{backup_path}.db"
            shutil.copy2(self.db_path, db_backup_path)
            print(f"  ✓ Database copied")
            
            # 2. Export data to JSON for easy inspection
            json_path = f"{backup_path}.json"
            self._export_to_json(json_path)
            print(f"  ✓ JSON export created")
            
            # 3. Export data to CSV
            csv_path = f"{backup_path}.csv"
            self._export_to_csv(csv_path)
            print(f"  ✓ CSV export created")
            
            # 4. Include log files if requested
            log_files = []
            if include_logs and os.path.exists('logs'):
                for log_file in os.listdir('logs'):
                    if log_file.endswith('.log'):
                        log_path = os.path.join('logs', log_file)
                        shutil.copy2(log_path, f"{backup_path}_{log_file}")
                        log_files.append(f"{backup_path}_{log_file}")
                if log_files:
                    print(f"  ✓ Log files included: {len(log_files)}")
            
            # 5. Create metadata file
            metadata = {
                'backup_time': datetime.now().isoformat(),
                'database_file': self.db_path,
                'total_records': self._get_record_count(),
                'backup_files': [
                    db_backup_path,
                    json_path,
                    csv_path
                ] + log_files,
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform
                }
            }
            
            metadata_path = f"{backup_path}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"  ✓ Metadata created")
            
            # 6. Compress if requested
            final_path = db_backup_path
            if compress:
                zip_path = f"{backup_path}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add all backup files to zip
                    for file_path in [db_backup_path, json_path, csv_path, metadata_path] + log_files:
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))
                            os.remove(file_path)  # Remove individual files after adding to zip
                
                final_path = zip_path
                print(f"  ✓ Backup compressed: {zip_path}")
            
            # 7. Create backup index
            self._update_backup_index(backup_name, final_path, metadata)
            
            print(f"\n[OK] Backup created successfully!")
            print(f"[FOLDER] Location: {final_path}")
            print(f"[DATA] Records backed up: {metadata['total_records']}")
            
            return final_path
            
        except Exception as e:
            print(f"[ERROR] Backup failed: {e}")
            return None
    
    def _export_to_json(self, json_path):
        """Export database to JSON format"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM access_events ORDER BY record_created_at")
        rows = cursor.fetchall()
        
        data = []
        for row in rows:
            row_dict = dict(row)
            # Convert datetime objects to strings
            for key, value in row_dict.items():
                if isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            data.append(row_dict)
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        conn.close()
    
    def _export_to_csv(self, csv_path):
        """Export database to CSV format"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM access_events ORDER BY record_created_at")
        rows = cursor.fetchall()
        
        if not rows:
            return
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(dict(rows[0]).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        
        conn.close()
    
    def _get_record_count(self):
        """Get total number of records in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM access_events")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def _update_backup_index(self, backup_name, backup_path, metadata):
        """Update backup index file"""
        index_file = os.path.join(self.backup_dir, 'backup_index.json')
        
        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {'backups': []}
        
        index['backups'].append({
            'name': backup_name,
            'path': backup_path,
            'timestamp': metadata['backup_time'],
            'records': metadata['total_records'],
            'size': os.path.getsize(backup_path) if os.path.exists(backup_path) else 0
        })
        
        # Keep only last 50 backups in index
        index['backups'] = sorted(
            index['backups'],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:50]
        
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2, default=str)
    
    def list_backups(self):
        """List all available backups"""
        index_file = os.path.join(self.backup_dir, 'backup_index.json')
        
        if not os.path.exists(index_file):
            print("No backups found")
            return []
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        print("\n[INFO] AVAILABLE BACKUPS")
        print("="*80)
        print(f"{'#':<3} {'Name':<25} {'Date':<20} {'Records':<10} {'Size':<12} {'Path'}")
        print("-"*80)
        
        for i, backup in enumerate(index.get('backups', []), 1):
            size_mb = backup.get('size', 0) / (1024 * 1024)
            timestamp = datetime.fromisoformat(backup['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{i:<3} {backup['name']:<25} {timestamp:<20} "
                  f"{backup['records']:<10} {size_mb:.2f} MB {'':<5} {backup['path']}")
        
        print("="*80)
        print(f"Total backups: {len(index.get('backups', []))}")
        
        return index.get('backups', [])
    
    def restore_backup(self, backup_path, verify=True):
        """
        Restore database from backup
        
        Args:
            backup_path: Path to backup file
            verify: Whether to verify backup before restoring
        """
        try:
            print(f"[RETRY] Restoring from backup: {backup_path}")
            
            # Check if backup exists
            if not os.path.exists(backup_path):
                print(f"[ERROR] Backup file not found: {backup_path}")
                return False
            
            # Extract if it's a zip file
            if backup_path.endswith('.zip'):
                print("  Extracting backup...")
                extract_dir = os.path.join(self.backup_dir, 'temp_restore')
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    zipf.extractall(extract_dir)
                
                # Find the database file in extracted files
                for file in os.listdir(extract_dir):
                    if file.endswith('.db'):
                        db_backup = os.path.join(extract_dir, file)
                        break
                else:
                    print("[ERROR] No database file found in backup")
                    return False
            else:
                db_backup = backup_path
            
            # Verify backup
            if verify and not self._verify_backup(db_backup):
                print("[ERROR] Backup verification failed")
                return False
            
            # Create backup of current database before restoring
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            current_backup = os.path.join(self.backup_dir, f'pre_restore_{timestamp}.db')
            shutil.copy2(self.db_path, current_backup)
            print(f"  ✓ Current database backed up to: {current_backup}")
            
            # Restore the database
            shutil.copy2(db_backup, self.db_path)
            print(f"  ✓ Database restored from: {db_backup}")
            
            # Clean up temporary files
            if backup_path.endswith('.zip'):
                shutil.rmtree(extract_dir)
            
            # Verify restoration
            restored_count = self._get_record_count()
            print(f"  ✓ Restoration verified: {restored_count} records")
            
            print("\n[OK] Database restored successfully!")
            print(f"[DATA] Records restored: {restored_count}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Restoration failed: {e}")
            return False
    
    def _verify_backup(self, backup_path):
        """Verify backup file integrity"""
        try:
            # Check if it's a valid SQLite database
            test_conn = sqlite3.connect(backup_path)
            cursor = test_conn.cursor()
            
            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='access_events'")
            if not cursor.fetchone():
                print("[ERROR] Backup missing required tables")
                test_conn.close()
                return False
            
            # Check record count
            cursor.execute("SELECT COUNT(*) FROM access_events")
            count = cursor.fetchone()[0]
            
            test_conn.close()
            
            print(f"  ✓ Backup verified: {count} records")
            return True
            
        except Exception as e:
            print(f"[ERROR] Backup verification error: {e}")
            return False
    
    def auto_cleanup(self, max_backups=10, max_age_days=30):
        """
        Automatically clean up old backups
        
        Args:
            max_backups: Maximum number of backups to keep
            max_age_days: Maximum age of backups to keep (days)
        """
        try:
            index_file = os.path.join(self.backup_dir, 'backup_index.json')
            
            if not os.path.exists(index_file):
                return 0
            
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            backups = index.get('backups', [])
            if not backups:
                return 0
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
            deleted_count = 0
            current_time = datetime.now()
            
            for i, backup in enumerate(backups):
                backup_time = datetime.fromisoformat(backup['timestamp'])
                age_days = (current_time - backup_time).days
                
                # Delete if too old or beyond max count
                if age_days > max_age_days or i >= max_backups:
                    backup_path = backup['path']
                    
                    if os.path.exists(backup_path):
                        os.remove(backup_path)
                        deleted_count += 1
                        print(f"  🗑️  Deleted old backup: {backup['name']} ({age_days} days old)")
            
            # Update index
            index['backups'] = backups[:max_backups]
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2, default=str)
            
            print(f"\n🗑️  Cleanup complete: Deleted {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            print(f"[ERROR] Auto-cleanup failed: {e}")
            return 0
    
    def schedule_backup(self, interval_hours=24):
        """
        Schedule automatic backups (run this from cron or task scheduler)
        """
        import schedule
        import time
        
        print(f"[TIMER] Scheduling automatic backups every {interval_hours} hours...")
        print("Press Ctrl+C to stop")
        
        def backup_job():
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled backup...")
            self.create_backup(compress=True, include_logs=True)
        
        schedule.every(interval_hours).hours.do(backup_job)
        
        # Run immediately
        backup_job()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("\n[STOP] Backup scheduler stopped")

def main():
    parser = argparse.ArgumentParser(description='Database Backup Utility')
    parser.add_argument('--backup', action='store_true', help='Create new backup')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--restore', type=str, help='Restore from backup file')
    parser.add_argument('--verify', type=str, help='Verify backup file')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old backups')
    parser.add_argument('--max-backups', type=int, default=10, help='Maximum backups to keep')
    parser.add_argument('--max-age', type=int, default=30, help='Maximum backup age in days')
    parser.add_argument('--schedule', type=int, help='Schedule backups every N hours')
    parser.add_argument('--no-compress', action='store_true', help='Disable compression')
    parser.add_argument('--no-logs', action='store_true', help='Exclude log files')
    
    args = parser.parse_args()
    
    backup = DatabaseBackup()
    
    if args.backup:
        backup.create_backup(
            compress=not args.no_compress,
            include_logs=not args.no_logs
        )
    
    elif args.list:
        backup.list_backups()
    
    elif args.restore:
        backup.restore_backup(args.restore)
    
    elif args.verify:
        if backup._verify_backup(args.verify):
            print("[OK] Backup verification successful")
        else:
            print("[ERROR] Backup verification failed")
    
    elif args.cleanup:
        backup.auto_cleanup(
            max_backups=args.max_backups,
            max_age_days=args.max_age
        )
    
    elif args.schedule:
        backup.schedule_backup(args.schedule)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()