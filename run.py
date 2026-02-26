#!/usr/bin/env python3
"""
Smart Lock System Launcher
Main entry point with comprehensive system initialization
"""
import os
import sys
import argparse
import logging
from datetime import datetime
import webbrowser
import threading
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Configure comprehensive logging system"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create loggers
    loggers = {
        'system': logging.getLogger('system'),
        'access': logging.getLogger('access'),
        'ai': logging.getLogger('ai'),
        'security': logging.getLogger('security'),
        'database': logging.getLogger('database')
    }
    
    # Configure handlers
    for name, logger in loggers.items():
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # Remove any existing handlers
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'{name}.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING if name == 'access' else logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.propagate = False
    
    return loggers

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask',
        'numpy',
        'requests',
        'sklearn',  # scikit-learn imports as sklearn
        'pandas'
    ]
    
    optional_packages = [
        'torch',
        'cv2',  # opencv-python
        'speechbrain',
        'PIL'  # Pillow
    ]
    
    missing = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing:
        print(f"[ERROR] Missing required dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"[WARNING]  Missing optional dependencies: {', '.join(missing_optional)}")
        print("Some features may not work. Run: pip install -r requirements.txt")
    
    return True

def initialize_database():
    """Initialize and seed database"""
    from event_logger import EventLogger
    
    print("[DATABASE] Initializing database...")
    
    try:
        logger = EventLogger()
        
        # Check if database has data
        stats = logger.get_statistics()
        
        if stats.get('total_events', 0) < 50:
            print("[DATABASE] Seeding database with initial data...")
            # You could call database_seeder here if needed
        
        print(f"[OK] Database ready ({stats.get('total_events', 0)} records)")
        return True
        
    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        return False

def initialize_ai_components():
    """Initialize AI and ML components"""
    print("[AI] Initializing AI components...")
    
    try:
        # Import and initialize components
        from ai_safety import GenAIAnalyzer
        from models.anomaly_detector import create_anomaly_detector
        from models.behavior_model import create_behavior_model
        from models.score_fusion import create_score_fusion
        
        # Initialize AI analyzer
        ai_analyzer = GenAIAnalyzer()
        
        # Test AI connection
        if ai_analyzer.api_key:
            print("  Testing AI API connection...")
            if ai_analyzer.test_connection():
                print("  [OK] AI API connected")
            else:
                print("  [WARNING]  AI API connection failed")
        else:
            print("  [WARNING]  No AI API key, running in simulation mode")
        
        # Initialize ML models
        print("  Loading ML models...")
        
        anomaly_detector = create_anomaly_detector()
        behavior_model = create_behavior_model()
        score_fusion = create_score_fusion()
        
        print("  [OK] AI/ML components initialized")
        return True
        
    except Exception as e:
        print(f"[ERROR] AI initialization failed: {e}")
        return False

def initialize_security():
    """Initialize security components"""
    print("[SECURITY] Initializing security...")
    
    try:
        from Configuration.config import Config
        from Configuration.security_config import SecurityConfig
        
        # Load configurations
        config = Config()
        security_config = SecurityConfig()
        
        # Validate configurations
        if not config.validate():
            print("  [WARNING]  Configuration validation warnings")
        
        # Check encryption keys
        print("  Checking encryption keys...")
        security_config.get_encryption_key()  # Will generate if missing
        
        print("  [OK] Security initialized")
        return True
        
    except Exception as e:
        print(f"[ERROR] Security initialization failed: {e}")
        return False

def start_web_interface(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask web interface"""
    print("[WEB] Starting web interface...")
    
    try:
        from app import app
        
        # Start in background thread
        def run_app():
            app.run(host=host, port=port, debug=debug, use_reloader=False)
        
        server_thread = threading.Thread(target=run_app, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        print(f"[OK] Web interface running at http://{host}:{port}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to start web interface: {e}")
        return False

def open_browser(host='0.0.0.0', port=5000):
    """Open web browser to the application"""
    url = f"http://{host}:{port}"
    
    print(f"[BROWSER] Opening browser to {url}")
    
    try:
        webbrowser.open(url)
        return True
    except:
        print("[WARNING]  Could not open browser automatically")
        return False

def run_system_tests():
    """Run comprehensive system tests"""
    print("[TESTS] Running system tests...")
    
    try:
        from Utilities.test_suite import run_all_tests
        
        success = run_all_tests()
        
        if success:
            print("[OK] All tests passed")
        else:
            print("[WARNING]  Some tests failed")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        return False

def backup_system():
    """Create system backup"""
    print("[BACKUP] Creating system backup...")
    
    try:
        from Utilities.backup_database import DatabaseBackup
        
        backup = DatabaseBackup()
        backup_path = backup.create_backup(compress=True, include_logs=True)
        
        if backup_path:
            print(f"[OK] Backup created: {backup_path}")
            return True
        else:
            print("[ERROR] Backup failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Backup failed: {e}")
        return False

def cleanup_system():
    """Clean up old files and logs"""
    print("[CLEANUP] Cleaning up system...")
    
    try:
        from Utilities.admin_tools import AdminTools
        
        admin = AdminTools()
        
        # Clean up old logs
        deleted = admin.cleanup_old_logs(days=30)
        print(f"  Deleted {deleted} old log entries")
        
        # Clean up temp files
        import tempfile
        import shutil
        
        temp_dir = tempfile.gettempdir()
        temp_files = [f for f in os.listdir(temp_dir) if f.startswith('temp_')]
        
        for temp_file in temp_files:
            try:
                os.remove(os.path.join(temp_dir, temp_file))
            except:
                pass
        
        print(f"  Cleaned up {len(temp_files)} temp files")
        return True
        
    except Exception as e:
        print(f"[WARNING]  Cleanup failed: {e}")
        return False

def print_system_banner():
    """Print system banner"""
    banner = """
    ==============================================================
                                                                  
      [LOCK] GENAI SMART LOCK SYSTEM - RESEARCH GRADE            
                                                                  
      Multi-Modal Authentication with Real-Time AI Analysis  
                                                                  
    ==============================================================
    """
    
    print(banner)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='GenAI Smart Lock System - Research Grade Access Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start system with default settings
  %(prog)s --host 127.0.0.1 --port 8080  # Custom host and port
  %(prog)s --no-browser --debug     # Start without browser, with debug
  %(prog)s --test                   # Run system tests only
  %(prog)s --backup                 # Create system backup
  %(prog)s --cleanup                # Clean up old files and logs
        """
    )
    
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser automatically')
    parser.add_argument('--test', action='store_true',
                       help='Run system tests and exit')
    parser.add_argument('--backup', action='store_true',
                       help='Create system backup and exit')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up system files and exit')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency checks (not recommended)')
    
    args = parser.parse_args()
    
    # Print banner
    print_system_banner()
    
    print(f"[STARTUP] Starting GenAI Smart Lock System")
    print(f"[TIME] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check for special modes
    if args.test:
        return run_system_tests()
    
    if args.backup:
        return backup_system()
    
    if args.cleanup:
        return cleanup_system()
    
    # Normal startup sequence
    print("[INIT] Initializing system...")
    
    # Setup logging
    loggers = setup_logging()
    loggers['system'].info(f"Starting system with args: {vars(args)}")
    
    # Check dependencies
    if not args.skip_deps and not check_dependencies():
        return 1
    
    # Initialize components
    components_ok = True
    
    components_ok &= initialize_database()
    components_ok &= initialize_ai_components()
    components_ok &= initialize_security()
    
    if not components_ok:
        loggers['system'].error("Component initialization failed")
        print("[ERROR] System initialization failed")
        return 1
    
    # Start web interface
    if not start_web_interface(args.host, args.port, args.debug):
        loggers['system'].error("Web interface failed to start")
        return 1
    
    # Open browser
    if not args.no_browser:
        open_browser(args.host if args.host != '0.0.0.0' else 'localhost', args.port)
    
    print("\n" + "="*60)
    print("[OK] SYSTEM READY")
    print("="*60)
    print(f"[WEB] Web Interface: http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    print(f"[INIT] Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"[FOLDER] Logs Directory: logs/")
    print(f"[DATABASE] Database: smart_lock_events.db")
    print("="*60)
    print("\n[INFO] Available Endpoints:")
    print("  /                 - Main dashboard")
    print("  /api/status       - System status")
    print("  /api/authenticate - Authentication endpoint")
    print("  /api/events       - Access events")
    print("  /api/stats        - System statistics")
    print("\n[ADMIN]  Admin Tools:")
    print("  python -m utilities.admin_tools --help")
    print("  python -m utilities.backup_database --help")
    print("\n[SHUTDOWN] Press Ctrl+C to stop the system")
    print("="*60)
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Shutting down system...")
        loggers['system'].info("System shutdown initiated")
        
        # Perform cleanup
        cleanup_system()
        
        print("[OK] System shutdown complete")
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Forced shutdown")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)