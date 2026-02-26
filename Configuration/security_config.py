"""
Security Configuration
Security-specific settings and validation for Smart Lock System
"""
import os
import hashlib
import secrets
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import json

class SecurityConfig:
    """Security configuration and validation"""
    
    # Default security settings
    DEFAULT_SECURITY = {
        # Lockout settings
        'LOCKOUT': {
            'MAX_FAILED_ATTEMPTS': 5,
            'LOCKOUT_DURATION_MINUTES': 30,
            'AUTO_RESET': True,
            'REQUIRE_ADMIN': False,
            'GRADUAL_INCREASE': True  # Increase lockout time with more failures
        },
        
        # PIN settings
        'PIN': {
            'MIN_LENGTH': 4,
            'MAX_LENGTH': 8,
            'ALLOW_SIMPLE_PATTERNS': False,
            'REQUIRE_DIGIT_VARIETY': True,
            'MAX_ATTEMPTS_PER_MINUTE': 10
        },
        
        # Biometric thresholds
        'BIOMETRIC': {
            'FACE_THRESHOLD': 0.7,
            'VOICE_THRESHOLD': 0.65,
            'MINIMUM_SCORE': 0.3,
            'REQUIRE_LIVENESS': False,
            'ALLOW_PARTIAL_MATCH': True
        },
        
        # Session security
        'SESSION': {
            'TIMEOUT_MINUTES': 30,
            'RENEWAL_INTERVAL': 15,
            'MAX_CONCURRENT_SESSIONS': 3,
            'ALLOW_REMEMBER_ME': False
        },
        
        # Rate limiting
        'RATE_LIMIT': {
            'REQUESTS_PER_MINUTE': 60,
            'AUTH_ATTEMPTS_PER_HOUR': 20,
            'IP_BLOCK_THRESHOLD': 100,
            'IP_BLOCK_DURATION_HOURS': 24
        },
        
        # Encryption
        'ENCRYPTION': {
            'ALGORITHM': 'AES-256-GCM',
            'KEY_ROTATION_DAYS': 90,
            'SALT_SIZE_BYTES': 32,
            'ITERATIONS': 100000
        },
        
        # Audit logging
        'AUDIT': {
            'ENABLED': True,
            'RETENTION_DAYS': 365,
            'LOG_FAILED_ATTEMPTS': True,
            'LOG_SUCCESSFUL_ACCESS': True,
            'LOG_ADMIN_ACTIONS': True
        },
        
        # Network security
        'NETWORK': {
            'REQUIRE_HTTPS': False,
            'ALLOWED_IPS': [],  # Empty = allow all
            'BLOCKED_IPS': [],
            'GEO_RESTRICTIONS': []  # Country codes
        },
        
        # Advanced security
        'ADVANCED': {
            'BEHAVIOR_ANALYSIS': True,
            'ANOMALY_DETECTION': True,
            'REAL_TIME_ALERTS': False,
            'AUTO_LEARNING': True,
            'RISK_BASED_AUTH': True
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize security configuration"""
        self.config_file = config_file or 'security.json'
        self.config = self.DEFAULT_SECURITY.copy()
        
        # Load custom security config if exists
        self.load_config()
        
        # Initialize security components
        self._init_security()
    
    def load_config(self):
        """Load security configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    custom_config = json.load(f)
                
                # Deep merge
                self._merge_configs(self.config, custom_config)
                print(f"[OK] Loaded security configuration from {self.config_file}")
        except Exception as e:
            print(f"[ERROR] Error loading security config: {e}")
    
    def _merge_configs(self, base: Dict, custom: Dict):
        """Recursively merge custom config into base"""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _init_security(self):
        """Initialize security components"""
        # Create necessary directories
        os.makedirs('security', exist_ok=True)
        os.makedirs('keys', exist_ok=True)
        
        # Initialize encryption keys if needed
        self._init_encryption_keys()
    
    def _init_encryption_keys(self):
        """Initialize or load encryption keys"""
        key_file = 'keys/encryption.key'
        salt_file = 'keys/encryption.salt'
        
        if not os.path.exists(key_file):
            # Generate new encryption key
            key = secrets.token_bytes(32)  # 256-bit key
            salt = secrets.token_bytes(self.config['ENCRYPTION']['SALT_SIZE_BYTES'])
            
            with open(key_file, 'wb') as f:
                f.write(key)
            
            with open(salt_file, 'wb') as f:
                f.write(salt)
            
            print("[KEY] Generated new encryption keys")
    
    def validate_pin(self, pin: str) -> Tuple[bool, str]:
        """
        Validate PIN according to security rules
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pin_config = self.config['PIN']
        
        # Check length
        if len(pin) < pin_config['MIN_LENGTH']:
            return False, f"PIN must be at least {pin_config['MIN_LENGTH']} digits"
        
        if len(pin) > pin_config['MAX_LENGTH']:
            return False, f"PIN must be at most {pin_config['MAX_LENGTH']} digits"
        
        # Check if it's all digits
        if not pin.isdigit():
            return False, "PIN must contain only digits"
        
        # Check for simple patterns
        if not pin_config['ALLOW_SIMPLE_PATTERNS']:
            # Check for repeated digits
            if len(set(pin)) == 1:
                return False, "PIN cannot be all the same digit"
            
            # Check for sequential digits
            if pin in ['1234', '4321', '5678', '8765']:
                return False, "PIN cannot be sequential digits"
        
        # Check for digit variety
        if pin_config['REQUIRE_DIGIT_VARIETY'] and len(set(pin)) < 3:
            return False, "PIN must contain at least 3 different digits"
        
        return True, "PIN is valid"
    
    def calculate_lockout_duration(self, failure_count: int) -> int:
        """
        Calculate lockout duration based on failure count
        
        Args:
            failure_count: Number of consecutive failures
            
        Returns:
            Lockout duration in minutes
        """
        lockout_config = self.config['LOCKOUT']
        base_duration = lockout_config['LOCKOUT_DURATION_MINUTES']
        
        if lockout_config['GRADUAL_INCREASE']:
            # Exponential increase: 30min, 60min, 120min, 240min, etc.
            return base_duration * (2 ** (failure_count - 1))
        else:
            return base_duration
    
    def should_lockout(self, failure_count: int) -> bool:
        """
        Determine if system should lockout based on failure count
        """
        return failure_count >= self.config['LOCKOUT']['MAX_FAILED_ATTEMPTS']
    
    def validate_biometric_scores(self, face_score: float, voice_score: float) -> Tuple[bool, Dict[str, str]]:
        """
        Validate biometric scores against thresholds
        
        Returns:
            Tuple of (is_valid, error_details)
        """
        bio_config = self.config['BIOMETRIC']
        errors = {}
        
        # Check minimum scores
        if face_score < bio_config['MINIMUM_SCORE']:
            errors['face'] = f"Face score below minimum ({bio_config['MINIMUM_SCORE']})"
        
        if voice_score < bio_config['MINIMUM_SCORE']:
            errors['voice'] = f"Voice score below minimum ({bio_config['MINIMUM_SCORE']})"
        
        # Check against thresholds
        if face_score < bio_config['FACE_THRESHOLD']:
            errors['face_threshold'] = f"Face score below threshold ({bio_config['FACE_THRESHOLD']})"
        
        if voice_score < bio_config['VOICE_THRESHOLD']:
            errors['voice_threshold'] = f"Voice score below threshold ({bio_config['VOICE_THRESHOLD']})"
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def calculate_risk_score(self, 
                           pin_valid: bool, 
                           face_score: float, 
                           voice_score: float,
                           failure_count: int,
                           access_time: Optional[datetime] = None) -> float:
        """
        Calculate overall risk score (0-1, where 0 = low risk, 1 = high risk)
        """
        risk_factors = []
        
        # PIN factor
        if not pin_valid:
            risk_factors.append(0.8)  # High risk for invalid PIN
        else:
            risk_factors.append(0.0)  # No risk for valid PIN
        
        # Face score factor (inverse: lower score = higher risk)
        face_risk = max(0, 1 - (face_score / self.config['BIOMETRIC']['FACE_THRESHOLD']))
        risk_factors.append(face_risk)
        
        # Voice score factor
        voice_risk = max(0, 1 - (voice_score / self.config['BIOMETRIC']['VOICE_THRESHOLD']))
        risk_factors.append(voice_risk)
        
        # Failure count factor
        max_failures = self.config['LOCKOUT']['MAX_FAILED_ATTEMPTS']
        failure_risk = min(1.0, failure_count / max_failures)
        risk_factors.append(failure_risk)
        
        # Time-based risk (if time provided)
        if access_time:
            hour = access_time.hour
            normal_start = 7
            normal_end = 22
            
            if hour < normal_start or hour > normal_end:
                time_risk = 0.3  # Moderate risk for unusual hours
            else:
                time_risk = 0.0  # Normal hours
            
            risk_factors.append(time_risk)
        
        # Calculate weighted average
        weights = [0.3, 0.25, 0.25, 0.2]  # Adjust weights as needed
        if len(risk_factors) == 5:  # Includes time risk
            weights = [0.25, 0.2, 0.2, 0.15, 0.2]
            weights = [w/sum(weights) for w in weights]  # Normalize
        
        risk_score = sum(w * f for w, f in zip(weights, risk_factors))
        
        return min(1.0, risk_score)
    
    def get_risk_level(self, risk_score: float) -> str:
        """
        Convert risk score to risk level
        """
        if risk_score < 0.3:
            return 'LOW'
        elif risk_score < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def should_audit(self, event_type: str) -> bool:
        """
        Check if event should be audited
        """
        audit_config = self.config['AUDIT']
        
        if not audit_config['ENABLED']:
            return False
        
        if event_type == 'FAILED_ATTEMPT':
            return audit_config['LOG_FAILED_ATTEMPTS']
        elif event_type == 'SUCCESSFUL_ACCESS':
            return audit_config['LOG_SUCCESSFUL_ACCESS']
        elif event_type == 'ADMIN_ACTION':
            return audit_config['LOG_ADMIN_ACTIONS']
        
        return True
    
    def check_rate_limit(self, ip_address: str, event_type: str) -> Tuple[bool, Optional[int]]:
        """
        Check rate limiting for IP address
        
        Returns:
            Tuple of (is_allowed, seconds_until_next_allowed)
        """
        # This is a simplified implementation
        # In production, use Redis or similar for distributed rate limiting
        
        rate_config = self.config['RATE_LIMIT']
        
        if event_type == 'AUTHENTICATION':
            limit = rate_config['AUTH_ATTEMPTS_PER_HOUR']
            window = 3600  # 1 hour
        else:
            limit = rate_config['REQUESTS_PER_MINUTE']
            window = 60  # 1 minute
        
        # Simplified: always allow for demo
        # In production, implement proper rate limiting
        
        return True, None
    
    def get_encryption_key(self) -> bytes:
        """Get encryption key"""
        key_file = 'keys/encryption.key'
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = secrets.token_bytes(32)
            os.makedirs('keys', exist_ok=True)
            
            with open(key_file, 'wb') as f:
                f.write(key)
            
            return key
    
    def encrypt_data(self, data: str) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt sensitive data
        
        Returns:
            Tuple of (ciphertext, tag, nonce)
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        key = self.get_encryption_key()
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)  # GCM nonce should be 12 bytes
        
        # Convert string to bytes
        data_bytes = data.encode('utf-8')
        
        # Encrypt
        ciphertext = aesgcm.encrypt(nonce, data_bytes, None)
        
        # For AES-GCM, tag is appended to ciphertext
        # Split ciphertext and tag (tag is last 16 bytes)
        tag = ciphertext[-16:]
        ciphertext = ciphertext[:-16]
        
        return ciphertext, tag, nonce
    
    def decrypt_data(self, ciphertext: bytes, tag: bytes, nonce: bytes) -> str:
        """Decrypt data"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        key = self.get_encryption_key()
        aesgcm = AESGCM(key)
        
        # Combine ciphertext and tag
        combined = ciphertext + tag
        
        # Decrypt
        plaintext = aesgcm.decrypt(nonce, combined, None)
        
        return plaintext.decode('utf-8')
    
    def save_config(self):
        """Save security configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"[OK] Security configuration saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving security config: {e}")
            return False
    
    def get_summary(self) -> Dict:
        """Get security configuration summary"""
        return {
            'lockout_threshold': self.config['LOCKOUT']['MAX_FAILED_ATTEMPTS'],
            'pin_requirements': {
                'min_length': self.config['PIN']['MIN_LENGTH'],
                'max_length': self.config['PIN']['MAX_LENGTH']
            },
            'biometric_thresholds': {
                'face': self.config['BIOMETRIC']['FACE_THRESHOLD'],
                'voice': self.config['BIOMETRIC']['VOICE_THRESHOLD']
            },
            'audit_enabled': self.config['AUDIT']['ENABLED'],
            'rate_limiting': {
                'auth_per_hour': self.config['RATE_LIMIT']['AUTH_ATTEMPTS_PER_HOUR'],
                'requests_per_minute': self.config['RATE_LIMIT']['REQUESTS_PER_MINUTE']
            },
            'advanced_features': {
                'behavior_analysis': self.config['ADVANCED']['BEHAVIOR_ANALYSIS'],
                'anomaly_detection': self.config['ADVANCED']['ANOMALY_DETECTION'],
                'risk_based_auth': self.config['ADVANCED']['RISK_BASED_AUTH']
            }
        }

# Global security configuration instance
_security_instance = None

def get_security_config(config_file: Optional[str] = None) -> SecurityConfig:
    """Get or create global security configuration"""
    global _security_instance
    
    if _security_instance is None:
        _security_instance = SecurityConfig(config_file)
    
    return _security_instance

if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Configuration Manager')
    parser.add_argument('--validate-pin', type=str, help='Validate a PIN')
    parser.add_argument('--calculate-risk', nargs=4, 
                       metavar=('PIN_VALID', 'FACE_SCORE', 'VOICE_SCORE', 'FAILURES'),
                       help='Calculate risk score')
    parser.add_argument('--show-summary', action='store_true', help='Show security summary')
    parser.add_argument('--test-encryption', type=str, help='Test encryption with sample text')
    parser.add_argument('--save', action='store_true', help='Save current configuration')
    
    args = parser.parse_args()
    
    security = get_security_config()
    
    if args.validate_pin:
        is_valid, message = security.validate_pin(args.validate_pin)
        print(f"PIN '{args.validate_pin}': {message}")
    
    elif args.calculate_risk:
        pin_valid = args.calculate_risk[0].lower() == 'true'
        face_score = float(args.calculate_risk[1])
        voice_score = float(args.calculate_risk[2])
        failures = int(args.calculate_risk[3])
        
        risk_score = security.calculate_risk_score(pin_valid, face_score, voice_score, failures)
        risk_level = security.get_risk_level(risk_score)
        
        print(f"Risk Score: {risk_score:.3f}")
        print(f"Risk Level: {risk_level}")
        
        # Also check biometric validation
        is_valid, errors = security.validate_biometric_scores(face_score, voice_score)
        if not is_valid:
            print("Biometric Errors:", errors)
    
    elif args.test_encryption:
        print(f"Original: {args.test_encryption}")
        
        # Encrypt
        ciphertext, tag, nonce = security.encrypt_data(args.test_encryption)
        print(f"Encrypted: {ciphertext.hex()[:50]}...")
        
        # Decrypt
        decrypted = security.decrypt_data(ciphertext, tag, nonce)
        print(f"Decrypted: {decrypted}")
        
        # Verify
        if decrypted == args.test_encryption:
            print("[OK] Encryption/decryption successful")
        else:
            print("[ERROR] Encryption/decryption failed")
    
    elif args.show_summary:
        summary = security.get_summary()
        print(json.dumps(summary, indent=2))
    
    elif args.save:
        security.save_config()
    
    else:
        parser.print_help()