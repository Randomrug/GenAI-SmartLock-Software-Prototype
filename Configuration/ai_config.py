"""
AI Configuration
Configuration for GenAI and machine learning models
"""
import os
import json
from typing import Dict, List, Optional, Any
import requests

class AIConfig:
    """AI and ML configuration management"""
    
    # Default AI configuration
    DEFAULT_CONFIG = {
        # AI Provider settings
        'PROVIDERS': {
            'openrouter': {
                'enabled': True,
                'api_url': 'https://openrouter.ai/api/v1/chat/completions',
                'models': [
                    'raptor/raptor-mini:preview',
                    'meta-llama/llama-3.3-70b-instruct:free',
                    'google/gemma-2-9b-it:free',
                    'mistralai/mistral-7b-instruct:free',
                    'qwen/qwen-2.5-32b-instruct:free',
                    'meta-llama/llama-3.1-8b-instruct:free'
                ],
                'default_model': 'raptor/raptor-mini:preview',
                'timeout': 30,
                'max_retries': 3
            },
            'openai': {
                'enabled': False,
                'api_url': 'https://api.openai.com/v1/chat/completions',
                'models': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                'default_model': 'gpt-3.5-turbo',
                'timeout': 30,
                'max_retries': 3
            },
            'anthropic': {
                'enabled': False,
                'api_url': 'https://api.anthropic.com/v1/messages',
                'models': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
                'default_model': 'claude-3-haiku',
                'timeout': 30,
                'max_retries': 3
            },
            'local': {
                'enabled': False,
                'api_url': 'http://localhost:11434/api/generate',
                'models': ['llama2', 'mistral', 'codellama'],
                'default_model': 'llama2',
                'timeout': 60,
                'max_retries': 1
            }
        },
        
        # Active provider (choose one)
        'ACTIVE_PROVIDER': 'openrouter',
        
        # Model parameters
        'MODEL_PARAMS': {
            'temperature': 0.3,
            'max_tokens': 300,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0,
            'stop': None
        },
        
        # Prompt engineering
        'PROMPT_TEMPLATES': {
            'security_analysis': '''You are an AI security analyst for a smart lock system.
Analyze this access attempt based on:

CURRENT ATTEMPT:
- Action: {action}
- Time: {datetime}
- PIN Valid: {pin_valid}
- Face Score: {face_score:.2f} (0-1)
- Voice Score: {voice_score:.2f} (0-1)
- Behavior Score: {behavior_score:.2f} (0-1)
- Final Score: {final_score:.2f} (0-1)
- Failed Attempts Streak: {failed_attempts}

HISTORICAL PATTERNS:
{historical_patterns}

RISK FACTORS:
{risk_factors}

ANALYSIS QUESTIONS:
1. Is this behavior consistent with historical patterns?
2. Are the biometric scores acceptable?
3. Is there a pattern of repeated failures?
4. What is the overall risk assessment?

Output JSON with: decision, risk_level, explanation''',
            
            'anomaly_detection': '''Detect anomalies in this access pattern:

Access Details:
- User: {user_id}
- Time: {access_time}
- Location: {location}
- Device: {device}
- Previous pattern: {previous_pattern}

Analyze for:
1. Geographic anomalies
2. Time-based anomalies
3. Device anomalies
4. Behavioral anomalies

Output: anomaly_score (0-1), confidence, reasons''',
            
            'explanation': '''Explain this security decision in user-friendly terms:

Decision: {decision}
Risk Level: {risk_level}
Technical Reason: {technical_reason}

Provide a clear, concise explanation that:
1. States the decision clearly
2. Explains the main factors
3. Suggests corrective actions if denied
4. Maintains security awareness'''
        },
        
        # Machine Learning models
        'ML_MODELS': {
            'behavior_analysis': {
                'enabled': True,
                'type': 'isolation_forest',
                'contamination': 0.1,
                'n_estimators': 100,
                'training_samples': 100
            },
            'score_fusion': {
                'enabled': True,
                'method': 'weighted_average',  # 'weighted_average', 'svm', 'neural_network'
                'weights': {'pin': 0.3, 'face': 0.35, 'voice': 0.35},
                'threshold': 0.7
            },
            'time_series': {
                'enabled': True,
                'window_size': 24,
                'prediction_horizon': 6,
                'seasonality': 'daily'
            }
        },
        
        # Training settings
        'TRAINING': {
            'auto_retrain_days': 7,
            'min_samples_for_training': 50,
            'validation_split': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001
        },
        
        # Caching
        'CACHING': {
            'enabled': True,
            'ttl_seconds': 300,
            'max_size': 1000,
            'strategy': 'lru'  # 'lru', 'lfu', 'fifo'
        },
        
        # Monitoring
        'MONITORING': {
            'log_ai_decisions': True,
            'track_accuracy': True,
            'alert_on_drift': True,
            'drift_threshold': 0.1,
            'performance_metrics': ['response_time', 'accuracy', 'confidence']
        },
        
        # Fallback settings
        'FALLBACK': {
            'enabled': True,
            'max_failures': 3,
            'cooldown_seconds': 60,
            'fallback_method': 'rule_based'  # 'rule_based', 'cached', 'random'
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize AI configuration"""
        self.config_file = config_file or 'ai_config.json'
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load custom config if exists
        self.load_config()
        
        # Load API keys from environment
        self._load_api_keys()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    custom_config = json.load(f)
                
                self._merge_configs(self.config, custom_config)
                print(f"[OK] Loaded AI configuration from {self.config_file}")
        except Exception as e:
            print(f"[ERROR] Error loading AI config: {e}")
    
    def _merge_configs(self, base: Dict, custom: Dict):
        """Recursively merge configurations"""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        # Load OpenRouter key
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if openrouter_key:
            self.config['PROVIDERS']['openrouter']['api_key'] = openrouter_key

        # Respect OPENROUTER_MODEL environment override (keep config in sync)
        openrouter_model = os.getenv('OPENROUTER_MODEL')
        if openrouter_model:
            # set as default model and ensure it's listed
            self.config['PROVIDERS']['openrouter']['default_model'] = openrouter_model
            if openrouter_model not in self.config['PROVIDERS']['openrouter']['models']:
                self.config['PROVIDERS']['openrouter']['models'].insert(0, openrouter_model)
        
        # Load OpenAI key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.config['PROVIDERS']['openai']['api_key'] = openai_key
        
        # Load Anthropic key
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.config['PROVIDERS']['anthropic']['api_key'] = anthropic_key
    
    def get_active_provider(self) -> Dict:
        """Get configuration for active provider"""
        provider_name = self.config['ACTIVE_PROVIDER']
        provider_config = self.config['PROVIDERS'].get(provider_name, {})
        
        if not provider_config.get('enabled', False):
            print(f"[WARNING]  Active provider '{provider_name}' is disabled")
        
        return {
            'name': provider_name,
            'config': provider_config,
            'model': provider_config.get('default_model', ''),
            'api_key': provider_config.get('api_key', '')
        }
    
    def get_model_params(self) -> Dict:
        """Get model parameters"""
        return self.config['MODEL_PARAMS']
    
    def get_prompt_template(self, template_name: str, **kwargs) -> str:
        """Get formatted prompt template"""
        template = self.config['PROMPT_TEMPLATES'].get(template_name, '')
        
        if template and kwargs:
            try:
                return template.format(**kwargs)
            except KeyError as e:
                print(f"[WARNING]  Missing key in prompt template: {e}")
        
        return template
    
    def get_ml_config(self, model_name: str) -> Dict:
        """Get ML model configuration"""
        return self.config['ML_MODELS'].get(model_name, {})
    
    def get_training_config(self) -> Dict:
        """Get training configuration"""
        return self.config['TRAINING']
    
    def get_caching_config(self) -> Dict:
        """Get caching configuration"""
        return self.config['CACHING']
    
    def get_monitoring_config(self) -> Dict:
        """Get monitoring configuration"""
        return self.config['MONITORING']
    
    def get_fallback_config(self) -> Dict:
        """Get fallback configuration"""
        return self.config['FALLBACK']
    
    def test_provider_connection(self, provider_name: Optional[str] = None) -> bool:
        """Test connection to AI provider"""
        if not provider_name:
            provider_name = self.config['ACTIVE_PROVIDER']
        
        provider_config = self.config['PROVIDERS'].get(provider_name)
        
        if not provider_config or not provider_config.get('enabled'):
            print(f"[ERROR] Provider '{provider_name}' is not enabled")
            return False
        
        api_key = provider_config.get('api_key')
        if not api_key:
            print(f"[ERROR] No API key found for provider '{provider_name}'")
            return False
        
        try:
            if provider_name == 'openrouter':
                return self._test_openrouter(api_key)
            elif provider_name == 'openai':
                return self._test_openai(api_key)
            elif provider_name == 'anthropic':
                return self._test_anthropic(api_key)
            elif provider_name == 'local':
                return self._test_local()
            else:
                print(f"[ERROR] Unknown provider: {provider_name}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Connection test failed: {e}")
            return False
    
    def _test_openrouter(self, api_key: str) -> bool:
        """Test OpenRouter connection"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config['PROVIDERS']['openrouter']['default_model'],
            "messages": [{"role": "user", "content": "Say 'connected'"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            self.config['PROVIDERS']['openrouter']['api_url'],
            headers=headers,
            json=payload,
            timeout=10
        )
        
        return response.status_code == 200
    
    def _test_openai(self, api_key: str) -> bool:
        """Test OpenAI connection"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Say 'connected'"}],
            "max_tokens": 10
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        return response.status_code == 200
    
    def _test_anthropic(self, api_key: str) -> bool:
        """Test Anthropic connection"""
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Say connected"}]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        return response.status_code == 200
    
    def _test_local(self) -> bool:
        """Test local LLM connection"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama2", "prompt": "Say connected", "stream": False},
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def get_available_models(self, provider_name: Optional[str] = None) -> List[str]:
        """Get available models for provider"""
        if not provider_name:
            provider_name = self.config['ACTIVE_PROVIDER']
        
        provider_config = self.config['PROVIDERS'].get(provider_name, {})
        return provider_config.get('models', [])
    
    def set_active_model(self, model_name: str, provider_name: Optional[str] = None):
        """Set active model for provider"""
        if not provider_name:
            provider_name = self.config['ACTIVE_PROVIDER']
        
        provider_config = self.config['PROVIDERS'].get(provider_name)
        if provider_config:
            available_models = provider_config.get('models', [])
            
            if model_name in available_models:
                provider_config['default_model'] = model_name
                print(f"[OK] Set active model to {model_name} for {provider_name}")
                return True
            else:
                print(f"[ERROR] Model {model_name} not available for {provider_name}")
                print(f"Available models: {', '.join(available_models)}")
                return False
        
        return False
    
    def enable_provider(self, provider_name: str) -> bool:
        """Enable an AI provider"""
        if provider_name in self.config['PROVIDERS']:
            self.config['PROVIDERS'][provider_name]['enabled'] = True
            print(f"[OK] Enabled provider: {provider_name}")
            return True
        else:
            print(f"[ERROR] Unknown provider: {provider_name}")
            return False
    
    def disable_provider(self, provider_name: str) -> bool:
        """Disable an AI provider"""
        if provider_name in self.config['PROVIDERS']:
            self.config['PROVIDERS'][provider_name]['enabled'] = False
            print(f"[OK] Disabled provider: {provider_name}")
            return True
        else:
            print(f"[ERROR] Unknown provider: {provider_name}")
            return False
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"[OK] AI configuration saved to {self.config_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Error saving AI config: {e}")
            return False
    
    def get_summary(self) -> Dict:
        """Get configuration summary"""
        active_provider = self.get_active_provider()
        
        return {
            'active_provider': {
                'name': active_provider['name'],
                'model': active_provider['model'],
                'enabled': self.config['PROVIDERS'][active_provider['name']]['enabled']
            },
            'model_parameters': self.config['MODEL_PARAMS'],
            'ml_models_enabled': {
                name: config['enabled']
                for name, config in self.config['ML_MODELS'].items()
            },
            'caching_enabled': self.config['CACHING']['enabled'],
            'fallback_enabled': self.config['FALLBACK']['enabled'],
            'monitoring': self.config['MONITORING']
        }

# Global AI configuration instance
_ai_instance = None

def get_ai_config(config_file: Optional[str] = None) -> AIConfig:
    """Get or create global AI configuration"""
    global _ai_instance
    
    if _ai_instance is None:
        _ai_instance = AIConfig(config_file)
    
    return _ai_instance

if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Configuration Manager')
    parser.add_argument('--test-connection', type=str, nargs='?', const='', 
                       help='Test provider connection (default: active provider)')
    parser.add_argument('--list-models', type=str, nargs='?', const='',
                       help='List available models for provider')
    parser.add_argument('--set-model', nargs=2, metavar=('MODEL', 'PROVIDER'),
                       help='Set active model for provider')
    parser.add_argument('--enable', type=str, help='Enable provider')
    parser.add_argument('--disable', type=str, help='Disable provider')
    parser.add_argument('--show-summary', action='store_true', help='Show configuration summary')
    parser.add_argument('--save', action='store_true', help='Save configuration')
    
    args = parser.parse_args()
    
    ai_config = get_ai_config()
    
    if args.test_connection:
        provider = args.test_connection if args.test_connection else None
        
        if ai_config.test_provider_connection(provider):
            print("[OK] Connection test successful")
        else:
            print("[ERROR] Connection test failed")
    
    elif args.list_models:
        provider = args.list_models if args.list_models else None
        models = ai_config.get_available_models(provider)
        
        print(f"Available models for {provider or 'active provider'}:")
        for model in models:
            print(f"  - {model}")
    
    elif args.set_model:
        model_name, provider = args.set_model
        success = ai_config.set_active_model(model_name, provider)
    
    elif args.enable:
        ai_config.enable_provider(args.enable)
    
    elif args.disable:
        ai_config.disable_provider(args.disable)
    
    elif args.show_summary:
        summary = ai_config.get_summary()
        print(json.dumps(summary, indent=2))
    
    elif args.save:
        ai_config.save_config()
    
    else:
        parser.print_help()