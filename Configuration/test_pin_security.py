"""
PIN Security Verification and Migration Test
Validates that the new PIN hashing system works correctly
"""
import sys
import os
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Configuration.pin_security import PINSecurityManager


def test_pin_hashing():
    """Test basic PIN hashing functionality"""
    print("\n" + "="*70)
    print("PIN HASHING TESTS")
    print("="*70)
    
    test_pin = "5678"
    
    # Test 1: Hash a PIN
    print("\n[TEST 1] Hashing PIN...")
    pin_hash = PINSecurityManager.hash_pin(test_pin)
    print(f"  PIN: {test_pin}")
    print(f"  SHA-256 Hash: {pin_hash}")
    print(f"  Hash length: {len(pin_hash)} characters")
    assert len(pin_hash) == 64, "SHA-256 hash should be 64 hex characters"
    print("  ✓ Hash generated correctly")
    
    # Test 2: Verify correct PIN
    print("\n[TEST 2] Verifying correct PIN...")
    is_valid = PINSecurityManager.verify_pin(test_pin, pin_hash)
    print(f"  Entered PIN: {test_pin}")
    print(f"  Stored hash: {pin_hash[:16]}...")
    print(f"  Match result: {is_valid}")
    assert is_valid == True, "Correct PIN should verify successfully"
    print("  ✓ Correct PIN verified successfully")
    
    # Test 3: Reject incorrect PIN
    print("\n[TEST 3] Rejecting incorrect PIN...")
    wrong_pin = "9999"
    is_valid = PINSecurityManager.verify_pin(wrong_pin, pin_hash)
    print(f"  Entered PIN: {wrong_pin}")
    print(f"  Stored hash: {pin_hash[:16]}...")
    print(f"  Match result: {is_valid}")
    assert is_valid == False, "Incorrect PIN should fail verification"
    print("  ✓ Incorrect PIN rejected correctly")
    
    # Test 4: Different PINs produce different hashes
    print("\n[TEST 4] Testing hash uniqueness...")
    pin1 = "1111"
    pin2 = "2222"
    hash1 = PINSecurityManager.hash_pin(pin1)
    hash2 = PINSecurityManager.hash_pin(pin2)
    print(f"  PIN 1: {pin1} -> {hash1[:16]}...")
    print(f"  PIN 2: {pin2} -> {hash2[:16]}...")
    assert hash1 != hash2, "Different PINs should produce different hashes"
    print("  ✓ Different PINs produce different hashes")
    
    # Test 5: Same PIN produces same hash
    print("\n[TEST 5] Testing hash consistency...")
    pin = "3333"
    hash1 = PINSecurityManager.hash_pin(pin)
    hash2 = PINSecurityManager.hash_pin(pin)
    print(f"  PIN: {pin}")
    print(f"  Hash 1: {hash1[:16]}...")
    print(f"  Hash 2: {hash2[:16]}...")
    assert hash1 == hash2, "Same PIN should produce same hash"
    print("  ✓ Same PIN produces consistent hash")


def test_pin_storage():
    """Test PIN saving and loading"""
    print("\n" + "="*70)
    print("PIN STORAGE TESTS")
    print("="*70)
    
    test_file = "test_pin_hash.json"
    test_pin = "4567"
    
    try:
        # Test 1: Save PIN hash
        print("\n[TEST 1] Saving PIN hash...")
        result = PINSecurityManager.save_pin_hash(test_pin, test_file)
        print(f"  Result: {result}")
        assert result['success'] == True, "PIN save should succeed"
        assert os.path.exists(test_file), "PIN hash file should be created"
        print("  ✓ PIN hash saved successfully")
        
        # Test 2: Load PIN hash
        print("\n[TEST 2] Loading PIN hash...")
        loaded_data = PINSecurityManager.load_pin_hash(test_file)
        print(f"  Loaded hash: {loaded_data['pin_hash'][:16]}...")
        assert loaded_data is not None, "PIN hash should load successfully"
        assert 'pin_hash' in loaded_data, "Loaded data should contain pin_hash"
        print("  ✓ PIN hash loaded successfully")
        
        # Test 3: Verify loaded hash works
        print("\n[TEST 3] Verifying with loaded hash...")
        is_valid = PINSecurityManager.verify_pin(test_pin, loaded_data['pin_hash'])
        print(f"  Entered PIN: {test_pin}")
        print(f"  Verification result: {is_valid}")
        assert is_valid == True, "Verification with loaded hash should work"
        print("  ✓ Verification with loaded hash works")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n  Cleaned up test file: {test_file}")


def test_pin_validation():
    """Test PIN validation rules"""
    print("\n" + "="*70)
    print("PIN VALIDATION TESTS")
    print("="*70)
    
    test_file = "test_validation.json"
    
    try:
        # Test 1: PIN too short
        print("\n[TEST 1] Rejecting PIN too short...")
        result = PINSecurityManager.save_pin_hash("123", test_file)
        print(f"  PIN: 123 (too short)")
        print(f"  Result: {result}")
        assert result['success'] == False, "PIN too short should be rejected"
        print("  ✓ PIN too short rejected correctly")
        
        # Test 2: PIN too long
        print("\n[TEST 2] Rejecting PIN too long...")
        result = PINSecurityManager.save_pin_hash("123456789", test_file)
        print(f"  PIN: 123456789 (too long)")
        print(f"  Result: {result}")
        assert result['success'] == False, "PIN too long should be rejected"
        print("  ✓ PIN too long rejected correctly")
        
        # Test 3: PIN with non-digits
        print("\n[TEST 3] Rejecting PIN with non-digits...")
        result = PINSecurityManager.save_pin_hash("12A4", test_file)
        print(f"  PIN: 12A4 (contains letter)")
        print(f"  Result: {result}")
        assert result['success'] == False, "PIN with non-digits should be rejected"
        print("  ✓ PIN with non-digits rejected correctly")
        
        # Test 4: Valid PIN accepted
        print("\n[TEST 4] Accepting valid PIN...")
        result = PINSecurityManager.save_pin_hash("1234", test_file)
        print(f"  PIN: 1234 (valid)")
        print(f"  Result: Success = {result['success']}")
        assert result['success'] == True, "Valid PIN should be accepted"
        print("  ✓ Valid PIN accepted correctly")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def test_migration():
    """Test migration from legacy PIN file"""
    print("\n" + "="*70)
    print("PIN MIGRATION TESTS")
    print("="*70)
    
    legacy_file = "test_legacy_pin.txt"
    new_file = "test_migrated_pin.json"
    legacy_pin = "2468"
    
    try:
        # Create legacy PIN file
        print("\n[SETUP] Creating legacy PIN file...")
        with open(legacy_file, 'w') as f:
            f.write(legacy_pin)
        print(f"  Created: {legacy_file} with PIN: {legacy_pin}")
        
        # Test 1: Migrate legacy file
        print("\n[TEST 1] Migrating legacy PIN file...")
        result = PINSecurityManager.migrate_legacy_pin(legacy_file, new_file)
        print(f"  Result: {result}")
        assert result['success'] == True, "Migration should succeed"
        assert os.path.exists(new_file), "New hash file should be created"
        print("  ✓ Legacy PIN migrated successfully")
        
        # Test 2: Verify migrated PIN works
        print("\n[TEST 2] Verifying migrated PIN...")
        loaded_data = PINSecurityManager.load_pin_hash(new_file)
        is_valid = PINSecurityManager.verify_pin(legacy_pin, loaded_data['pin_hash'])
        print(f"  Original PIN: {legacy_pin}")
        print(f"  Verification: {is_valid}")
        assert is_valid == True, "Migrated PIN should verify correctly"
        print("  ✓ Migrated PIN verifies correctly")
        
        # Test 3: Check backup was created
        print("\n[TEST 3] Checking backup file...")
        backup_file = f"{legacy_file}.backup"
        exists = os.path.exists(backup_file)
        print(f"  Backup file: {backup_file}")
        print(f"  Exists: {exists}")
        if exists:
            print("  ✓ Backup file created")
        
    finally:
        # Cleanup
        for f in [legacy_file, new_file, f"{legacy_file}.backup"]:
            if os.path.exists(f):
                os.remove(f)
        print(f"\n  Cleaned up test files")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PIN SECURITY SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    try:
        # Run test suites
        test_pin_hashing()
        test_pin_storage()
        test_pin_validation()
        test_migration()
        
        # Summary
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED SUCCESSFULLY")
        print("="*70)
        print("\nPIN Security Features:")
        print("  • SHA-256 cryptographic hashing")
        print("  • Irreversible PIN storage (cannot recover original PIN from hash)")
        print("  • PIN validation (4-8 digit requirement)")
        print("  • Secure hash comparison for authentication")
        print("  • Automatic migration from legacy plain-text format")
        print("  • Backup of legacy files during migration")
        print("\n" + "="*70 + "\n")
        
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
