"""
System-wide Configuration
Centralized configuration management for Smart Lock System
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class Config:
    """Central configuration manager"""
    
    # Default configuration
    DEFAULT_CONFIG = {
        # Server settings
        'SERVER': {
            'HOST': '0.0.0.0',
            'PORT': 5000,
            'DEBUG': True,
            'SECRET_KEY': 'dev-secret-key-change-in-production',
            'MAX_CONTENT_LENGTH_MB': 16
        },
        
        # Database settings
        'DATABASE': {
            'PATH': 'smart_lock_events.db',
            'BACKUP_DIR': 'backups',
            'MAX_BACKUPS': 10,
            'BACKUP_AGE_DAYS': 30,
            'AUTO_BACKUP_HOURS': 24
        },
        
        # Authentication settings
        'AUTHENTICATION': {
            'PIN_FILE': 'pin.txt',
            'DEFAULT_PIN': '1234',
            'PIN_MIN_LENGTH': 4,
            'PIN_MAX_LENGTH': 8,
            'FACE_THRESHOLD': 0.7,
            'VOICE_THRESHOLD': 0.65,
            'WEIGHTS': {
                'PIN': 0.3,
                'FACE': 0.35,
                'VOICE': 0.35
            }
        },
        
        # Security settings
        'SECURITY': {
            'MAX_FAILED_ATTEMPTS': 5,
            'LOCKOUT_DURATION_MINUTES': 30,
            'LOCKOUT_AUTO_RESET': True,
            'REQUIRE_ADMIN_RESET': False,
            'SESSION_TIMEOUT_MINUTES': 30
        },
        
        # AI Settings
        'AI': {
            'PROVIDER': 'openrouter',  # 'openrouter', 'openai', 'simulation'
            'MODEL': 'meta-llama/llama-3.3-70b-instruct:free',
            'TEMPERATURE': 0.3,
            'MAX_TOKENS': 300,
            'TIMEOUT_SECONDS': 30
        },
        
        # Logging settings
        'LOGGING': {
            'LEVEL': 'INFO',
            'DIRECTORY': 'logs',
            'MAX_LOG_SIZE_MB': 10,
            'BACKUP_COUNT': 5,
            'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        
        # File paths
        'PATHS': {
            'FACE_DB': 'embeddings/face_db.pkl',
            'VOICE_TEMPLATE': 'templates/my_voice_template.pt',
            'UPLOAD_FOLDER': 'temp_uploads',
            'ASSETS_DIR': 'assets'
        },
        
        # Behavior analysis
        'BEHAVIOR': {
            'NORMAL_HOURS_START': 7,
            'NORMAL_HOURS_END': 22,
            'SUSPICIOUS_HOURS': [0, 1, 2, 3, 4, 5],
            'LEARNING_WINDOW_DAYS': 30,
            'MIN_SAMPLES_FOR_PATTERN': 10
        },
        
        # Performance settings
        'PERFORMANCE': {
            'CACHE_ENABLED': True,
            'CACHE_TTL_SECONDS': 300,
            'QUERY_LIMIT': 1000,
            'CONNECTION_POOL_SIZE': 5
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration"""
        self.config_file = config_file or 'config.json'
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load custom configuration if exists
        self.load_config()
        
        # Set up logging
        self.setup_logging()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    custom_config = json.load(f)
                
                # Deep merge custom config with defaults
                self._merge_configs(self.config, custom_config)
                print(f"[OK] Loaded configuration from {self.config_file}")
            else:
                print(f"[WARNING]  No config file found, using defaults. Create {self.config_file} for custom settings.")
        except Exception as e:
            print(f"[ERROR] Error loading config file: {e}")
    
    def _merge_configs(self, base: Dict, custom: Dict):
        """Recursively merge custom config into base config"""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file) or '.', exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"[OK] Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving config file: {e}")
            return False
    
    def setup_logging(self):
        """Configure logging system"""
        log_config = self.config['LOGGING']
        
        # Create log directory
        log_dir = log_config['DIRECTORY']
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config['LEVEL']),
            format=log_config['FORMAT'],
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'system.log')),
                logging.StreamHandler()
            ]
        )
        
        # Add file rotation for access logs
        from logging.handlers import RotatingFileHandler
        
        access_handler = RotatingFileHandler(
            os.path.join(log_dir, 'access.log'),
            maxBytes=log_config['MAX_LOG_SIZE_MB'] * 1024 * 1024,
            backupCount=log_config['BACKUP_COUNT']
        )
        access_handler.setFormatter(logging.Formatter(log_config['FORMAT']))
        
        access_logger = logging.getLogger('access')
        access_logger.addHandler(access_handler)
        access_logger.propagate = False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_server_config(self) -> Dict:
        """Get server configuration"""
        return self.config['SERVER']
    
    def get_database_config(self) -> Dict:
        """Get database configuration"""
        return self.config['DATABASE']
    
    def get_auth_config(self) -> Dict:
        """Get authentication configuration"""
        return self.config['AUTHENTICATION']
    
    def get_security_config(self) -> Dict:
        """Get security configuration"""
        return self.config['SECURITY']
    
    def get_ai_config(self) -> Dict:
        """Get AI configuration"""
        return self.config['AI']
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration"""
        return self.config['LOGGING']
    
    def get_paths(self) -> Dict:
        """Get path configuration"""
        return self.config['PATHS']
    
    def get_behavior_config(self) -> Dict:
        """Get behavior analysis configuration"""
        return self.config['BEHAVIOR']
    
    def get_performance_config(self) -> Dict:
        """Get performance configuration"""
        return self.config['PERFORMANCE']
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Validate authentication weights sum to 1
        weights = self.get_auth_config()['WEIGHTS']
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(f"Authentication weights must sum to 1.0 (current: {weight_sum})")
        
        # Validate thresholds are between 0 and 1
        if not 0 <= self.get_auth_config()['FACE_THRESHOLD'] <= 1:
            errors.append("Face threshold must be between 0 and 1")
        
        if not 0 <= self.get_auth_config()['VOICE_THRESHOLD'] <= 1:
            errors.append("Voice threshold must be between 0 and 1")
        
        # Validate security settings
        if self.get_security_config()['MAX_FAILED_ATTEMPTS'] <= 0:
            errors.append("Max failed attempts must be positive")
        
        # Validate paths exist or can be created
        paths = self.get_paths()
        for key, path in paths.items():
            if key.endswith('_DIR') or key.endswith('_FOLDER'):
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {path}: {e}")
        
        if errors:
            print("[ERROR] Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("[OK] Configuration validated successfully")
        return True
    
    def create_default_config_file(self):
        """Create a default configuration file"""
        if os.path.exists(self.config_file):
            print(f"[WARNING]  Config file already exists: {self.config_file}")
            return False
        
        self.save_config()
        print(f"[OK] Created default config file: {self.config_file}")
        return True
    
    def reload(self):
        """Reload configuration from file"""
        self.load_config()
        self.setup_logging()
        print("[RETRY] Configuration reloaded")

# Global configuration instance
_config_instance = None

def get_config(config_file: Optional[str] = None) -> Config:
    """Get or create global configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance

# Convenience functions
def get(key: str, default: Any = None) -> Any:
    """Get configuration value (shortcut)"""
    return get_config().get(key, default)

def set(key: str, value: Any):
    """Set configuration value (shortcut)"""
    get_config().set(key, value)

def get_server() -> Dict:
    """Get server config (shortcut)"""
    return get_config().get_server_config()

def get_database() -> Dict:
    """Get database config (shortcut)"""
    return get_config().get_database_config()

def get_auth() -> Dict:
    """Get auth config (shortcut)"""
    return get_config().get_auth_config()

def get_security() -> Dict:
    """Get security config (shortcut)"""
    return get_config().get_security_config()

def get_ai() -> Dict:
    """Get AI config (shortcut)"""
    return get_config().get_ai_config()

def get_paths() -> Dict:
    """Get paths config (shortcut)"""
    return get_config().get_paths()

if __name__ == "__main__":
    # Command line interface for configuration
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Manager')
    parser.add_argument('--create-default', action='store_true', help='Create default config file')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--get', type=str, help='Get configuration value (dot notation)')
    parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set configuration value')
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--config-file', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    config = get_config(args.config_file)
    
    if args.create_default:
        config.create_default_config_file()
    
    elif args.validate:
        config.validate()
    
    elif args.get:
        value = config.get(args.get)
        print(f"{args.get} = {value}")
    
    elif args.set:
        key, value_str = args.set
        
        # Try to parse value as appropriate type
        try:
            value = json.loads(value_str)
        except json.JSONDecodeError:
            # If not JSON, keep as string or convert to number if possible
            if value_str.lower() in ['true', 'false']:
                value = value_str.lower() == 'true'
            elif value_str.isdigit():
                value = int(value_str)
            elif value_str.replace('.', '', 1).isdigit() and value_str.count('.') <= 1:
                value = float(value_str)
            else:
                value = value_str
        
        config.set(key, value)
        print(f"Set {key} = {value}")
        config.save_config()
    
    elif args.show:
        print(json.dumps(config.config, indent=2, default=str))
    
    else:
        parser.print_help()