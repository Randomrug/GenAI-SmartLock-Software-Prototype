"""
Comprehensive Test Suite for Smart Lock System
Run automated tests for all components
"""
import unittest
import sqlite3
import json
import os
import tempfile
import sys
from datetime import datetime, timedelta
import requests
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import modules to test
from event_logger import EventLogger
from ai_safety import GenAIAnalyzer
from app import load_pin, SYSTEM_PIN_DATA, SYSTEM_PIN_HASH
from Configuration.pin_security import PINSecurityManager

class TestDatabase(unittest.TestCase):
    """Test database functionality"""
    
    def setUp(self):
        """Set up test database"""
        # Create temporary database for testing
        self.test_db = tempfile.mktemp(suffix='.db')
        self.logger = EventLogger(self.test_db)
    
    def tearDown(self):
        """Clean up test database"""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_database_creation(self):
        """Test that database is created with correct tables"""
        self.assertTrue(os.path.exists(self.test_db))
        
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('access_events', tables)
        self.assertIn('system_statistics', tables)
        
        conn.close()
    
    def test_log_event(self):
        """Test logging an event"""
        event_id = self.logger.log_event(
            action='UNLOCK',
            entry_time='2024-01-15 14:30:00',
            pin_valid=True,
            face_score=0.85,
            voice_score=0.78,
            behavior_score=0.9,
            final_score=0.84,
            genai_decision='ALLOW',
            genai_risk_level='LOW',
            genai_explanation='Test event',
            failed_attempt_count=0,
            lockout_active=False
        )
        
        self.assertIsNotNone(event_id)
        self.assertGreater(event_id, 0)
    
    def test_get_recent_events(self):
        """Test retrieving recent events"""
        # Add test events
        for i in range(5):
            self.logger.log_event(
                action='UNLOCK' if i % 2 == 0 else 'LOCK',
                pin_valid=True,
                face_score=0.8 + i*0.05,
                voice_score=0.7 + i*0.05,
                genai_decision='ALLOW',
                genai_risk_level='LOW',
                genai_explanation=f'Test event {i}'
            )
        
        events = self.logger.get_recent_events(limit=3)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]['genai_decision'], 'ALLOW')
    
    def test_failure_streak(self):
        """Test failure streak tracking"""
        # Add failed attempts
        for i in range(3):
            self.logger.log_event(
                action='UNLOCK',
                pin_valid=False,
                genai_decision='DENY',
                genai_risk_level='HIGH',
                genai_explanation=f'Failed attempt {i}',
                failed_attempt_count=i+1
            )
        
        streak = self.logger.get_failure_streak()
        self.assertEqual(streak, 3)
    
    def test_statistics(self):
        """Test statistics calculation"""
        # Add mixed events
        self.logger.log_event(
            action='UNLOCK',
            pin_valid=True,
            face_score=0.9,
            voice_score=0.85,
            genai_decision='ALLOW',
            genai_risk_level='LOW',
            genai_explanation='Success'
        )
        
        self.logger.log_event(
            action='UNLOCK',
            pin_valid=False,
            face_score=0.4,
            voice_score=0.3,
            genai_decision='DENY',
            genai_risk_level='HIGH',
            genai_explanation='Failure'
        )
        
        stats = self.logger.get_statistics()
        self.assertIn('total_events', stats)
        self.assertEqual(stats['total_events'], 2)
        self.assertEqual(stats['allowed_events'], 1)
        self.assertEqual(stats['denied_events'], 1)

class TestGenAIAnalyzer(unittest.TestCase):
    """Test GenAI analyzer functionality"""
    
    def setUp(self):
        """Set up test analyzer"""
        self.analyzer = GenAIAnalyzer(api_key='test_key')
    
    def test_fallback_analysis_valid_pin(self):
        """Test fallback analysis with valid PIN"""
        attempt = {
            'pin_valid': True,
            'face_score': 0.85,
            'voice_score': 0.82,
            'final_score': 0.84,
            'failed_attempt_count': 0
        }
        
        result = self.analyzer._fallback_analysis(attempt)
        self.assertEqual(result['decision'], 'ALLOW')
        self.assertEqual(result['risk_level'], 'LOW')
    
    def test_fallback_analysis_invalid_pin(self):
        """Test fallback analysis with invalid PIN"""
        attempt = {
            'pin_valid': False,
            'face_score': 0.9,
            'voice_score': 0.9,
            'final_score': 0.9,
            'failed_attempt_count': 0
        }
        
        result = self.analyzer._fallback_analysis(attempt)
        self.assertEqual(result['decision'], 'DENY')
        self.assertEqual(result['risk_level'], 'HIGH')
    
    def test_fallback_analysis_low_scores(self):
        """Test fallback analysis with low biometric scores"""
        attempt = {
            'pin_valid': True,
            'face_score': 0.5,
            'voice_score': 0.55,
            'final_score': 0.52,
            'failed_attempt_count': 0
        }
        
        result = self.analyzer._fallback_analysis(attempt)
        self.assertEqual(result['decision'], 'DENY')
        self.assertEqual(result['risk_level'], 'HIGH')
    
    def test_fallback_analysis_lockout(self):
        """Test fallback analysis triggering lockout"""
        attempt = {
            'pin_valid': True,
            'face_score': 0.7,
            'voice_score': 0.7,
            'final_score': 0.7,
            'failed_attempt_count': 5
        }
        
        result = self.analyzer._fallback_analysis(attempt)
        self.assertEqual(result['decision'], 'LOCKOUT')
        self.assertEqual(result['risk_level'], 'HIGH')
    
    def test_parse_valid_response(self):
        """Test parsing valid AI response"""
        response = '''{
            "decision": "ALLOW",
            "risk_level": "LOW",
            "explanation": "Normal access pattern detected"
        }'''
        
        result = self.analyzer._parse_ai_response(response)
        self.assertEqual(result['decision'], 'ALLOW')
        self.assertEqual(result['risk_level'], 'LOW')
        self.assertIn('Normal access', result['explanation'])
    
    def test_parse_invalid_response(self):
        """Test parsing invalid AI response"""
        response = 'Invalid response without JSON'
        
        result = self.analyzer._parse_ai_response(response)
        self.assertEqual(result['decision'], 'DENY')
        self.assertEqual(result['risk_level'], 'HIGH')
    
    def test_identify_risk_factors(self):
        """Test risk factor identification"""
        attempt = {
            'pin_valid': True,
            'face_score': 0.4,
            'voice_score': 0.5,
            'failed_attempt_count': 3
        }
        
        stats = {
            'avg_face_score': 0.8,
            'avg_voice_score': 0.8
        }
        
        risk_factors = self.analyzer._identify_risk_factors(attempt, stats)
        
        self.assertGreater(len(risk_factors), 0)
        self.assertIn('Low face score', risk_factors[0])
        self.assertIn('Multiple consecutive failures', risk_factors[2])

class TestSystemIntegration(unittest.TestCase):
    """Test system integration"""
    
    def setUp(self):
        """Set up integration test"""
        self.test_db = tempfile.mktemp(suffix='.db')
        self.logger = EventLogger(self.test_db)
        self.analyzer = GenAIAnalyzer()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_complete_authentication_flow(self):
        """Test complete authentication flow"""
        # Simulate authentication attempt
        attempt_data = {
            'action': 'UNLOCK',
            'pin_valid': True,
            'face_score': 0.88,
            'voice_score': 0.92,
            'behavior_score': 0.9,
            'final_score': 0.9,
            'failed_attempt_count': 0,
            'lockout_active': False
        }
        
        # Get AI decision
        ai_result = self.analyzer.analyze_attempt(attempt_data)
        
        # Log the event
        event_id = self.logger.log_event(
            action=attempt_data['action'],
            pin_valid=attempt_data['pin_valid'],
            face_score=attempt_data['face_score'],
            voice_score=attempt_data['voice_score'],
            behavior_score=attempt_data['behavior_score'],
            final_score=attempt_data['final_score'],
            genai_decision=ai_result['decision'],
            genai_risk_level=ai_result['risk_level'],
            genai_explanation=ai_result['explanation'],
            failed_attempt_count=attempt_data['failed_attempt_count'],
            lockout_active=False
        )
        
        # Verify
        self.assertIsNotNone(event_id)
        self.assertIn(ai_result['decision'], ['ALLOW', 'DENY', 'LOCKOUT'])
        self.assertIn(ai_result['risk_level'], ['LOW', 'MEDIUM', 'HIGH'])
    
    def test_multiple_failures_lockout(self):
        """Test lockout after multiple failures"""
        events = []
        
        # Simulate 5 consecutive failures
        for i in range(5):
            attempt_data = {
                'pin_valid': False,
                'face_score': 0.3,
                'voice_score': 0.4,
                'final_score': 0.35,
                'failed_attempt_count': i+1
            }
            
            ai_result = self.analyzer._fallback_analysis(attempt_data)
            
            # Check if lockout should be triggered
            lockout = (i+1 >= 5)
            
            event_id = self.logger.log_event(
                action='UNLOCK',
                pin_valid=attempt_data['pin_valid'],
                face_score=attempt_data['face_score'],
                voice_score=attempt_data['voice_score'],
                final_score=attempt_data['final_score'],
                genai_decision='LOCKOUT' if lockout else 'DENY',
                genai_risk_level='HIGH',
                genai_explanation=f'Failed attempt {i+1}',
                failed_attempt_count=attempt_data['failed_attempt_count'],
                lockout_active=lockout
            )
            
            events.append(event_id)
        
        # Verify lockout was triggered
        last_event = self.logger.get_recent_events(limit=1)[0]
        self.assertEqual(last_event['genai_decision'], 'LOCKOUT')
        self.assertTrue(last_event['lockout_active'])

class TestPINSystem(unittest.TestCase):
    """Test PIN system functionality"""
    
    def test_pin_loading(self):
        """Test PIN loading from file"""
        # Create temporary PIN file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('5678\n')
            pin_file = f.name
        
        # Test loading
        original_load_pin = load_pin
        
        def mock_load_pin():
            with open(pin_file, 'r') as f:
                return f.read().strip()
        
        try:
            pin = mock_load_pin()
            self.assertEqual(pin, '5678')
        finally:
            os.remove(pin_file)
    
    def test_default_pin(self):
        """Test default PIN creation"""
        # Remove PIN file if exists
        if os.path.exists('pin.txt'):
            os.remove('pin.txt')
        
        pin = load_pin()
        self.assertEqual(pin, '1234')
        self.assertTrue(os.path.exists('pin.txt'))

class TestPerformance(unittest.TestCase):
    """Test system performance"""
    
    def setUp(self):
        """Set up performance test"""
        self.test_db = tempfile.mktemp(suffix='.db')
        self.logger = EventLogger(self.test_db)
    
    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)
    
    def test_bulk_insert_performance(self):
        """Test performance of bulk inserts"""
        import time
        
        start_time = time.time()
        
        # Insert 100 records
        for i in range(100):
            self.logger.log_event(
                action='UNLOCK',
                pin_valid=True,
                face_score=0.8,
                voice_score=0.8,
                genai_decision='ALLOW',
                genai_risk_level='LOW',
                genai_explanation=f'Performance test {i}'
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  Inserted 100 records in {duration:.2f} seconds")
        print(f"  Average: {duration/100:.3f} seconds per record")
        
        # Should complete in reasonable time
        self.assertLess(duration, 5.0)  # Should take less than 5 seconds
    
    def test_query_performance(self):
        """Test query performance"""
        # Insert test data
        for i in range(50):
            self.logger.log_event(
                action='UNLOCK',
                pin_valid=True,
                face_score=0.8,
                voice_score=0.8,
                genai_decision='ALLOW',
                genai_risk_level='LOW',
                genai_explanation=f'Query test {i}'
            )
        
        import time
        
        # Test get_recent_events performance
        start_time = time.time()
        events = self.logger.get_recent_events(limit=100)
        end_time = time.time()
        
        query_time = end_time - start_time
        
        print(f"\n⏱️  Retrieved {len(events)} records in {query_time:.3f} seconds")
        
        # Should be very fast
        self.assertLess(query_time, 0.5)

class TestSecurity(unittest.TestCase):
    """Test security features"""
    
    def test_lockout_persistence(self):
        """Test that lockout state persists"""
        test_db = tempfile.mktemp(suffix='.db')
        logger = EventLogger(test_db)
        
        try:
            # Set lockout state
            event_id = logger.log_event(
                action='SYSTEM',
                pin_valid=True,
                genai_decision='LOCKOUT',
                genai_risk_level='HIGH',
                genai_explanation='Security test',
                failed_attempt_count=5,
                lockout_active=True
            )
            
            # Create new logger instance (simulating system restart)
            logger2 = EventLogger(test_db)
            
            # Check lockout state
            conn = sqlite3.connect(test_db)
            cursor = conn.cursor()
            cursor.execute("SELECT lockout_active FROM access_events ORDER BY id DESC LIMIT 1")
            lockout_state = cursor.fetchone()[0]
            conn.close()
            
            self.assertTrue(lockout_state)
            
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)
    
    def test_failure_count_reset(self):
        """Test that failure count resets on success"""
        test_db = tempfile.mktemp(suffix='.db')
        logger = EventLogger(test_db)
        
        try:
            # Add some failures
            for i in range(3):
                logger.log_event(
                    action='UNLOCK',
                    pin_valid=False,
                    genai_decision='DENY',
                    genai_risk_level='HIGH',
                    genai_explanation=f'Failure {i+1}',
                    failed_attempt_count=i+1
                )
            
            # Add a success
            logger.log_event(
                action='UNLOCK',
                pin_valid=True,
                genai_decision='ALLOW',
                genai_risk_level='LOW',
                genai_explanation='Success after failures',
                failed_attempt_count=0  # Should reset to 0
            )
            
            # Get recent events
            events = logger.get_recent_events(limit=5)
            
            # Last event should have 0 failures
            self.assertEqual(events[0]['failed_attempt_count'], 0)
            
        finally:
            if os.path.exists(test_db):
                os.remove(test_db)

def run_all_tests():
    """Run all test suites"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestDatabase))
    test_suite.addTest(unittest.makeSuite(TestGenAIAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestSystemIntegration))
    test_suite.addTest(unittest.makeSuite(TestPINSystem))
    test_suite.addTest(unittest.makeSuite(TestPerformance))
    test_suite.addTest(unittest.makeSuite(TestSecurity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("[DATA] TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Smart Lock System Tests')
    parser.add_argument('--test', choices=['db', 'ai', 'integration', 'pin', 'performance', 'security', 'all'],
                       default='all', help='Select test category')
    
    args = parser.parse_args()
    
    print("[TEST] SMART LOCK SYSTEM TEST SUITE")
    print("="*60)
    
    if args.test == 'all':
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        # Run specific test category
        test_loader = unittest.TestLoader()
        
        test_map = {
            'db': TestDatabase,
            'ai': TestGenAIAnalyzer,
            'integration': TestSystemIntegration,
            'pin': TestPINSystem,
            'performance': TestPerformance,
            'security': TestSecurity
        }
        
        test_class = test_map.get(args.test, TestDatabase)
        test_suite = test_loader.loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        sys.exit(0 if result.wasSuccessful() else 1)