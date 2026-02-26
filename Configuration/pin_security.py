"""
PIN Security Module - Using SHA-256 Hashing for Secure PIN Storage
Pins are stored as irreversible hashes. Only matching hashes grant access.
"""
import hashlib
import json
import os
from pathlib import Path


class PINSecurityManager:
    """Manages secure PIN storage and verification using SHA-256 hashing"""
    
    # File to store hashed PIN
    PIN_HASH_FILE = 'pin_hash.json'
    LEGACY_PIN_FILE = 'pin.txt'
    
    # Hash algorithm and iterations for robustness
    HASH_ALGORITHM = 'sha256'
    
    @staticmethod
    def hash_pin(pin: str) -> str:
        """
        Hash a PIN using SHA-256
        
        Args:
            pin: Plain text PIN to hash
            
        Returns:
            Hex-encoded SHA-256 hash of the PIN
        """
        if not isinstance(pin, str):
            pin = str(pin)
        
        # Create SHA-256 hash
        hash_object = hashlib.sha256(pin.encode())
        return hash_object.hexdigest()
    
    @staticmethod
    def verify_pin(entered_pin: str, stored_hash: str) -> bool:
        """
        Verify that entered PIN matches stored hash
        
        Args:
            entered_pin: PIN entered by user
            stored_hash: Stored SHA-256 hash
            
        Returns:
            True if PIN matches hash, False otherwise
        """
        if not isinstance(entered_pin, str):
            entered_pin = str(entered_pin)
        
        entered_hash = PINSecurityManager.hash_pin(entered_pin)
        return entered_hash == stored_hash
    
    @staticmethod
    def save_pin_hash(pin: str, file_path: str = None) -> dict:
        """
        Hash and save PIN to secure storage
        
        Args:
            pin: Plain text PIN to save
            file_path: Path to PIN hash file (default: pin_hash.json)
            
        Returns:
            Dictionary with success status and hash info
        """
        if file_path is None:
            file_path = PINSecurityManager.PIN_HASH_FILE
        
        try:
            # Validate PIN
            if not isinstance(pin, str) or not pin:
                return {'success': False, 'error': 'PIN must be a non-empty string'}
            
            if len(pin) < 4:
                return {'success': False, 'error': 'PIN must be at least 4 digits'}
            
            if len(pin) > 8:
                return {'success': False, 'error': 'PIN must be at most 8 digits'}
            
            if not pin.isdigit():
                return {'success': False, 'error': 'PIN must contain only digits'}
            
            # Hash the PIN
            pin_hash = PINSecurityManager.hash_pin(pin)
            
            # Create data structure
            pin_data = {
                'pin_hash': pin_hash,
                'algorithm': PINSecurityManager.HASH_ALGORITHM,
                'version': '1.0',
                'created_at': str(Path(file_path).stat().st_mtime) if os.path.exists(file_path) else None
            }
            
            # Save to JSON file
            with open(file_path, 'w') as f:
                json.dump(pin_data, f, indent=2)
            
            print(f"[OK] PIN hash saved successfully to {file_path}")
            return {
                'success': True,
                'message': 'PIN saved securely',
                'hash': pin_hash,
                'file': file_path
            }
        
        except Exception as e:
            print(f"[ERROR] Failed to save PIN hash: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def load_pin_hash(file_path: str = None) -> dict:
        """
        Load PIN hash from secure storage
        
        Args:
            file_path: Path to PIN hash file
            
        Returns:
            Dictionary with hash and metadata, or default if not found
        """
        if file_path is None:
            file_path = PINSecurityManager.PIN_HASH_FILE
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"[OK] Loaded PIN hash from {file_path}")
                return data
            else:
                print(f"[WARNING] PIN hash file not found: {file_path}")
                return None
        
        except Exception as e:
            print(f"[ERROR] Failed to load PIN hash: {e}")
            return None
    
    @staticmethod
    def migrate_legacy_pin(legacy_file: str = None, new_file: str = None) -> dict:
        """
        Migrate plain-text PIN from legacy file to hashed format
        
        Args:
            legacy_file: Path to old plain-text PIN file
            new_file: Path to new hash file
            
        Returns:
            Migration result dictionary
        """
        if legacy_file is None:
            legacy_file = PINSecurityManager.LEGACY_PIN_FILE
        if new_file is None:
            new_file = PINSecurityManager.PIN_HASH_FILE
        
        try:
            # Check if legacy file exists
            if not os.path.exists(legacy_file):
                return {
                    'success': False,
                    'error': f'Legacy PIN file not found: {legacy_file}'
                }
            
            # Check if new file already exists (avoid overwriting)
            if os.path.exists(new_file):
                return {
                    'success': False,
                    'error': f'Hash file already exists: {new_file}. Migration might have been done already.'
                }
            
            # Read plain-text PIN
            with open(legacy_file, 'r') as f:
                plain_pin = f.read().strip()
            
            if not plain_pin:
                return {'success': False, 'error': 'Legacy PIN file is empty'}
            
            # Save as hash
            result = PINSecurityManager.save_pin_hash(plain_pin, new_file)
            
            if result['success']:
                print(f"[SUCCESS] Migrated PIN from {legacy_file} to {new_file}")
                # Optionally backup legacy file
                backup_file = f"{legacy_file}.backup"
                try:
                    with open(legacy_file, 'r') as f_in:
                        with open(backup_file, 'w') as f_out:
                            f_out.write(f_in.read())
                    print(f"[OK] Legacy PIN backed up to {backup_file}")
                except Exception as e:
                    print(f"[WARNING] Could not backup legacy file: {e}")
            
            return result
        
        except Exception as e:
            print(f"[ERROR] Migration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def init_pin_system(legacy_file: str = None, new_file: str = None) -> dict:
        """
        Initialize PIN system - handles migration if needed
        
        Args:
            legacy_file: Path to legacy plain-text PIN file
            new_file: Path to new hash file
            
        Returns:
            PIN hash data if system is ready, None otherwise
        """
        if legacy_file is None:
            legacy_file = PINSecurityManager.LEGACY_PIN_FILE
        if new_file is None:
            new_file = PINSecurityManager.PIN_HASH_FILE
        
        # Try to load existing hash file
        pin_data = PINSecurityManager.load_pin_hash(new_file)
        
        if pin_data and 'pin_hash' in pin_data:
            print(f"[OK] PIN system ready - using hash file {new_file}")
            return pin_data
        
        # Try to migrate from legacy file
        if os.path.exists(legacy_file):
            print(f"[INFO] Legacy PIN file detected - migrating to secure hash format...")
            migration_result = PINSecurityManager.migrate_legacy_pin(legacy_file, new_file)
            
            if migration_result['success']:
                # Load and return the migrated hash
                return PINSecurityManager.load_pin_hash(new_file)
            else:
                print(f"[ERROR] Migration failed: {migration_result.get('error')}")
                return None
        
        # No PIN system exists - create default
        print(f"[NOTE] Creating default PIN system...")
        default_pin = "1234"
        result = PINSecurityManager.save_pin_hash(default_pin, new_file)
        
        if result['success']:
            pin_data = PINSecurityManager.load_pin_hash(new_file)
            print(f"[NOTE] Default PIN '1234' hash created at {new_file}")
            return pin_data
        else:
            print(f"[ERROR] Failed to create default PIN: {result.get('error')}")
            return None
