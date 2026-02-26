"""
Admin Tools for Smart Lock System
Administrative functions for system management
"""
import sqlite3
import json
import csv
from datetime import datetime, timedelta
import os
import sys
import argparse

class AdminTools:
    def __init__(self, db_path='smart_lock_events.db'):
        self.db_path = db_path
    
    def reset_lockout(self, admin_key='admin123'):
        """Reset all lockout states"""
        if admin_key != 'admin123':
            print("[ERROR] Invalid admin key")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE access_events 
                SET lockout_active = 0 
                WHERE lockout_active = 1
            """)
            
            rows_affected = cursor.rowcount
            
            if rows_affected > 0:
                # Log the reset
                cursor.execute('''
                    INSERT INTO access_events (
                        action, pin_valid, face_score, voice_score,
                        behavior_score, final_score, genai_decision,
                        genai_risk_level, genai_explanation, failed_attempt_count,
                        lockout_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    'SYSTEM', True, 1.0, 1.0, 1.0, 1.0,
                    'ALLOW', 'LOW', 'System lockout reset by administrator',
                    0, False
                ))
            
            conn.commit()
            conn.close()
            
            print(f"[OK] Reset lockout on {rows_affected} records")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error resetting lockout: {e}")
            return False
    
    def view_logs(self, limit=50, filter_type=None, export_format=None):
        """View system logs"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM access_events"
            params = []
            
            if filter_type == 'lockout':
                query += " WHERE lockout_active = 1"
            elif filter_type == 'denied':
                query += " WHERE genai_decision = 'DENY'"
            elif filter_type == 'allowed':
                query += " WHERE genai_decision = 'ALLOW'"
            elif filter_type == 'recent':
                query += " WHERE record_created_at >= datetime('now', '-7 days')"
            
            query += " ORDER BY record_created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if export_format == 'json':
                return self._export_json(rows)
            elif export_format == 'csv':
                return self._export_csv(rows)
            else:
                self._display_logs(rows)
            
            conn.close()
            
        except Exception as e:
            print(f"[ERROR] Error viewing logs: {e}")
    
    def _display_logs(self, rows):
        """Display logs in formatted table"""
        if not rows:
            print("No logs found")
            return
        
        print("\n[INFO] ACCESS LOGS")
        print("="*120)
        print(f"{'ID':<5} {'Time':<20} {'Action':<8} {'PIN':<5} {'Face':<6} {'Voice':<6} {'Decision':<10} {'Risk':<8} {'Failures':<8}")
        print("-"*120)
        
        for row in rows:
            time_str = row['record_created_at'][:19] if row['record_created_at'] else 'N/A'
            pin = '✓' if row['pin_valid'] else '✗'
            face = f"{row['face_score']:.2f}" if row['face_score'] else 'N/A'
            voice = f"{row['voice_score']:.2f}" if row['voice_score'] else 'N/A'
            
            # Color coding for decisions
            decision = row['genai_decision']
            if decision == 'ALLOW':
                decision_display = f"[OK] {decision}"
            elif decision == 'DENY':
                decision_display = f"[ERROR] {decision}"
            elif decision == 'LOCKOUT':
                decision_display = f"[SECURE] {decision}"
            else:
                decision_display = decision
            
            print(f"{row['id']:<5} {time_str:<20} {row['action']:<8} {pin:<5} {face:<6} {voice:<6} {decision_display:<10} {row['genai_risk_level']:<8} {row['failed_attempt_count']:<8}")
        
        print("="*120)
        print(f"Total records: {len(rows)}")
    
    def _export_json(self, rows):
        """Export logs to JSON"""
        data = [dict(row) for row in rows]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"logs/export_{timestamp}.json"
        
        os.makedirs('logs', exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"[OK] Exported {len(data)} records to {filename}")
        return filename
    
    def _export_csv(self, rows):
        """Export logs to CSV"""
        if not rows:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"logs/export_{timestamp}.csv"
        
        os.makedirs('logs', exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(dict(rows[0]).keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        
        print(f"[OK] Exported {len(rows)} records to {filename}")
        return filename
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM access_events")
            stats['total'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'ALLOW'")
            stats['allowed'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'DENY'")
            stats['denied'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
            stats['lockouts'] = cursor.fetchone()[0]
            
            # Time-based stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as today_count,
                    SUM(CASE WHEN genai_decision = 'ALLOW' THEN 1 ELSE 0 END) as today_allowed
                FROM access_events 
                WHERE date(record_created_at) = date('now')
            """)
            today_stats = cursor.fetchone()
            stats['today'] = {
                'total': today_stats[0] or 0,
                'allowed': today_stats[1] or 0
            }
            
            # Failure patterns
            cursor.execute("""
                SELECT 
                    AVG(failed_attempt_count) as avg_failures,
                    MAX(failed_attempt_count) as max_failures,
                    SUM(CASE WHEN failed_attempt_count >= 3 THEN 1 ELSE 0 END) as multiple_failures
                FROM access_events
            """)
            failure_stats = cursor.fetchone()
            stats['failures'] = {
                'average': failure_stats[0] or 0,
                'maximum': failure_stats[1] or 0,
                'multiple': failure_stats[2] or 0
            }
            
            # Score statistics
            cursor.execute("""
                SELECT 
                    AVG(face_score) as avg_face,
                    STDDEV(face_score) as std_face,
                    AVG(voice_score) as avg_voice,
                    STDDEV(voice_score) as std_voice,
                    AVG(final_score) as avg_final
                FROM access_events
            """)
            score_stats = cursor.fetchone()
            stats['scores'] = {
                'face': {
                    'average': score_stats[0] or 0,
                    'stddev': score_stats[1] or 0
                },
                'voice': {
                    'average': score_stats[2] or 0,
                    'stddev': score_stats[3] or 0
                },
                'final': {
                    'average': score_stats[4] or 0
                }
            }
            
            # Behavioral patterns
            cursor.execute("""
                SELECT strftime('%H', record_created_at) as hour, COUNT(*) as count
                FROM access_events 
                WHERE genai_decision = 'ALLOW'
                GROUP BY strftime('%H', record_created_at)
                ORDER BY count DESC
                LIMIT 3
            """)
            
            peak_hours = cursor.fetchall()
            stats['peak_hours'] = [{'hour': int(h), 'count': c} for h, c in peak_hours]
            
            conn.close()
            
            # Display statistics
            self._display_stats(stats)
            
            return stats
            
        except Exception as e:
            print(f"[ERROR] Error getting statistics: {e}")
            return {}
    
    def _display_stats(self, stats):
        """Display statistics in formatted way"""
        print("\n[DATA] SYSTEM STATISTICS")
        print("="*60)
        
        print(f"\n[STATS] OVERVIEW:")
        print(f"  Total Attempts: {stats.get('total', 0)}")
        print(f"  Allowed: {stats.get('allowed', 0)} ({stats.get('allowed', 0)/stats.get('total', 1)*100:.1f}%)")
        print(f"  Denied: {stats.get('denied', 0)} ({stats.get('denied', 0)/stats.get('total', 1)*100:.1f}%)")
        print(f"  Lockouts: {stats.get('lockouts', 0)}")
        
        print(f"\n[DATE] TODAY'S ACTIVITY:")
        today = stats.get('today', {})
        print(f"  Attempts: {today.get('total', 0)}")
        print(f"  Success Rate: {today.get('allowed', 0)/max(today.get('total', 1), 1)*100:.1f}%")
        
        print(f"\nFAILURE ANALYSIS:")
        failures = stats.get('failures', {})
        print(f"  Average Failures: {failures.get('average', 0):.2f}")
        print(f"  Maximum Failures: {failures.get('maximum', 0)}")
        print(f"  Multiple Failures (≥3): {failures.get('multiple', 0)}")
        
        print(f"\n[TARGET] SCORE STATISTICS:")
        scores = stats.get('scores', {})
        print(f"  Face: {scores.get('face', {}).get('average', 0):.3f} ± {scores.get('face', {}).get('stddev', 0):.3f}")
        print(f"  Voice: {scores.get('voice', {}).get('average', 0):.3f} ± {scores.get('voice', {}).get('stddev', 0):.3f}")
        print(f"  Final: {scores.get('final', {}).get('average', 0):.3f}")
        
        print(f"\n[TIMER] PEAK ACCESS HOURS (Successful):")
        for hour_data in stats.get('peak_hours', []):
            print(f"  {hour_data['hour']:02d}:00 - {hour_data['count']} attempts")
        
        print("="*60)
    
    def cleanup_old_logs(self, days=30):
        """Clean up logs older than specified days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM access_events 
                WHERE record_created_at < datetime('now', ?)
            """, (f'-{days} days',))
            
            rows_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"🗑️  Deleted {rows_deleted} records older than {days} days")
            return rows_deleted
            
        except Exception as e:
            print(f"[ERROR] Error cleaning up logs: {e}")
            return 0
    
    def backup_database(self, backup_dir='backups'):
        """Create a backup of the database"""
        import shutil
        import time
        
        try:
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(backup_dir, f'smart_lock_backup_{timestamp}.db')
            
            shutil.copy2(self.db_path, backup_path)
            
            print(f"[BACKUP] Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"[ERROR] Error backing up database: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Admin Tools for Smart Lock System')
    parser.add_argument('--reset-lockout', action='store_true', help='Reset system lockout')
    parser.add_argument('--view-logs', type=int, nargs='?', const=50, help='View logs (optional limit)')
    parser.add_argument('--filter', choices=['lockout', 'denied', 'allowed', 'recent'], help='Filter logs')
    parser.add_argument('--export', choices=['json', 'csv'], help='Export format')
    parser.add_argument('--stats', action='store_true', help='Show system statistics')
    parser.add_argument('--cleanup', type=int, help='Cleanup logs older than N days')
    parser.add_argument('--backup', action='store_true', help='Backup database')
    parser.add_argument('--admin-key', default='admin123', help='Admin key (default: admin123)')
    
    args = parser.parse_args()
    
    admin = AdminTools()
    
    if args.reset_lockout:
        admin.reset_lockout(args.admin_key)
    
    if args.view_logs:
        admin.view_logs(limit=args.view_logs, filter_type=args.filter, export_format=args.export)
    
    if args.stats:
        admin.get_system_stats()
    
    if args.cleanup:
        admin.cleanup_old_logs(args.cleanup)
    
    if args.backup:
        admin.backup_database()
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()