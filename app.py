

"""
Main Flask application for GenAI Smart Lock System
Real GenAI API Integration with OpenRouter
"""
import os
import sys
import json
import sqlite3
import tempfile
import threading
import uuid
import shutil
import pickle
import random
import string
from datetime import datetime, timedelta
import torch
import numpy as np
from flask import Flask, jsonify, render_template, request, Response
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pydub import AudioSegment
from queue import Queue

# Add custom modules to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(BASE_DIR, "VoiceRecognition")
FACENET_DIR = os.path.join(BASE_DIR, "FaceNet")
ALERT_DIR = os.path.join(BASE_DIR, "Alert_system")

# Add to Python path for imports
for p in (VOICE_DIR, FACENET_DIR, ALERT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load environment variables from .env file
load_dotenv()

# Import our modules
from event_logger import EventLogger
from ai_safety import GenAIAnalyzer
from Alert_system.sms import send_sms
from Alert_system.call import make_call
from Alert_system.send_email import send_email
from Alert_system.send_email import RECEIVER_EMAIL
from Configuration.pin_security import PINSecurityManager

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global variables
db_logger = EventLogger()
ai_analyzer = GenAIAnalyzer()
FACE_THRESHOLD = 0.9
VOICE_THRESHOLD = 0.65

# Biometric modules (initialized later)
face_module = None
voice_module = None
progress_queues = {}

# Image tracking for investigation
captured_images = []  # Keep track of last captured images for investigation
INVESTIGATION_IMAGES_DIR = os.path.join(BASE_DIR, "temp_uploads", "investigation")
os.makedirs(INVESTIGATION_IMAGES_DIR, exist_ok=True)

def store_investigation_image(face_path):
    """Store captured face image for investigation and keep last 2."""
    global captured_images
    try:
        if not face_path or not os.path.exists(face_path):
            return []
        investigation_img_path = os.path.join(
            INVESTIGATION_IMAGES_DIR,
            f"attempt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )
        shutil.copy2(face_path, investigation_img_path)
        captured_images.append(investigation_img_path)
        # Keep only last 2 images
        if len(captured_images) > 2:
            old_img = captured_images.pop(0)
            if os.path.exists(old_img):
                os.remove(old_img)
        return list(captured_images)
    except Exception as e:
        print(f"[WARNING]  Error saving investigation image: {e}")
        return list(captured_images)

# In-memory OTP store for password change flows: {email: {'otp': str, 'expires_at': datetime, 'verified': bool}}
password_otps = {}

# Thresholds
VOICE_WEIGHT = 0.5
FACE_WEIGHT = 0.5
COMBINED_THRESHOLD = 0.75

def init_lockout_otp_store():
    """Initialize persistent OTP store for lockout reset."""
    try:
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS lockout_otps (
                email TEXT PRIMARY KEY,
                otp TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                verified INTEGER DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to init lockout OTP store: {e}")

init_lockout_otp_store()

def init_system_settings():
    """Initialize system settings storage (e.g., mode)."""
    try:
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mode_otps (
                email TEXT PRIMARY KEY,
                otp TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                verified INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('''
            INSERT OR IGNORE INTO system_settings (key, value)
            VALUES ('mode', 'in_town')
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to init system settings: {e}")

def get_system_mode():
    try:
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM system_settings WHERE key='mode'")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else 'in_town'
    except Exception as e:
        print(f"[ERROR] Failed to get system mode: {e}")
        return 'in_town'

def set_system_mode(mode_value: str):
    try:
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_settings (key, value)
            VALUES ('mode', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        ''', (mode_value,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to set system mode: {e}")
        return False

init_system_settings()

def load_pin():
    """Load PIN hash using secure PIN security manager"""
    try:
        # Initialize PIN system (handles migration from plain-text if needed)
        pin_data = PINSecurityManager.init_pin_system()
        
        if pin_data and 'pin_hash' in pin_data:
            print(f"[OK] PIN security system initialized")
            return pin_data
        else:
            print(f"[ERROR] Failed to initialize PIN system")
            # Return default hash for 1234
            default_hash = PINSecurityManager.hash_pin("1234")
            return {'pin_hash': default_hash, 'algorithm': 'sha256'}
    except Exception as e: 	 
        print(f"[ERROR] Error loading PIN: {e}")
        # Fallback: return hash of default PIN
        default_hash = PINSecurityManager.hash_pin("1234")
        return {'pin_hash': default_hash, 'algorithm': 'sha256'}


SYSTEM_PIN_DATA = load_pin()
SYSTEM_PIN_HASH = SYSTEM_PIN_DATA.get('pin_hash', '') if SYSTEM_PIN_DATA else ''
@app.route('/api/check_modules')
def check_modules():
    """Check if modules are loading properly"""
    print("\n" + "="*50)
    print("[SEARCH] CHECKING MODULES")
    print("="*50)
    
    results = {
        'face_module': 'NOT LOADED',
        'voice_module': 'NOT LOADED',
        'face_db_size': 0,
        'voice_simulated': True
    }
    
    # Check FaceNet
    if face_module:
        results['face_module'] = 'LOADED'
        results['face_db_size'] = len(face_module.get('face_db', {}))
        print(f"[USER] Face module: LOADED ({results['face_db_size']} faces in DB)")
        
        # Test import
        try:
            from FaceNet.face_embedder import get_face_embedding
            print("[OK] face_embedder imports successfully")
        except ImportError as e:
            print(f"[ERROR] face_embedder import failed: {e}")
            
    else:
        print("[ERROR] Face module: NOT LOADED")
        print(f"[FOLDER] Checking FaceNet directory: {FACENET_DIR}")
        print(f"[FOLDER] Files in FaceNet: {os.listdir(FACENET_DIR) if os.path.exists(FACENET_DIR) else 'Directory not found'}")
    
    # Check VoiceRecognition
    if voice_module:
        results['voice_module'] = 'LOADED'
        results['voice_simulated'] = voice_module.get('simulated', True)
        print(f"[VOICE] Voice module: LOADED (Simulated: {results['voice_simulated']})")
    else:
        print("[ERROR] Voice module: NOT LOADED")
        print(f"[FOLDER] Checking VoiceRecognition directory: {VOICE_DIR}")
        print(f"[FOLDER] Files in VoiceRecognition: {os.listdir(VOICE_DIR) if os.path.exists(VOICE_DIR) else 'Directory not found'}")
    
    print("="*50)
    return jsonify(results)


@app.route('/reports')
def reports():
    """Return heatmap, trend data, and stats for reports - supports owner filtering"""
    from flask import request
    try:
        owner_filter = request.args.get('owner', 'overall').lower()
        
        conn = sqlite3.connect(db_logger.db_path)
        cursor = conn.cursor()

        # Build WHERE clause based on owner filter - use parameterized query
        if owner_filter not in ['overall', '']:
            where_condition = f"owner = '{owner_filter}'"
        else:
            where_condition = "1=1"

        # Heatmap (weekday): successful login events (IN + ALLOW) per day_of_week and hour
        cursor.execute(f'''
            SELECT strftime('%w', COALESCE(event_datetime, record_created_at)) as dow,
                   strftime('%H', COALESCE(event_datetime, record_created_at)) as hour,
                   COUNT(*) as cnt
            FROM access_events
            WHERE {where_condition}
              AND action = 'IN'
              AND genai_decision = 'ALLOW'
            GROUP BY dow, hour
        ''')
        rows = cursor.fetchall()

        days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        heatmap = {d: [0]*24 for d in days}
        for dow,hour,cnt in rows:
            try:
                h = int(hour)
            except Exception:
                h = 0
            try:
                dow_int = int(dow)
            except Exception:
                dow_int = None
            if dow_int is None:
                continue
            # SQLite %w: 0=Sunday..6=Saturday
            day_name = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][dow_int]
            if day_name in heatmap:
                heatmap[day_name][h] = cnt


        # Trends: last 14 days total counts and high/medium/low counts
        cursor.execute(f'''
            SELECT date(COALESCE(event_datetime, record_created_at)) as day,
                   COUNT(*) as total,
                   SUM(CASE WHEN genai_risk_level='HIGH' THEN 1 ELSE 0 END) as high,
                   SUM(CASE WHEN genai_risk_level='MEDIUM' THEN 1 ELSE 0 END) as medium,
                   SUM(CASE WHEN genai_risk_level='LOW' THEN 1 ELSE 0 END) as low
            FROM access_events
            WHERE {where_condition}
            AND COALESCE(event_datetime, record_created_at) >= date('now','-14 day')
            GROUP BY date(COALESCE(event_datetime, record_created_at))
            ORDER BY day ASC
        ''')
        trend_rows = cursor.fetchall()
        trend_map = {}
        for day,total,high,medium,low in trend_rows:
            trend_map[day] = {
                'day': day,
                'total': total or 0,
                'high': high or 0,
                'medium': medium or 0,
                'low': low or 0
            }

        # Fill missing days so charts don't look empty
        from datetime import datetime, timedelta
        trends = []
        today = datetime.now().date()
        for i in range(13, -1, -1):
            d = (today - timedelta(days=i)).isoformat()
            trends.append(trend_map.get(d, {'day': d, 'total': 0, 'high': 0, 'medium': 0, 'low': 0}))

        # Decision distribution (ALLOW/DENY/LOCKOUT)
        cursor.execute(f'''
            SELECT genai_decision, COUNT(*) as cnt
            FROM access_events
            WHERE {where_condition}
            GROUP BY genai_decision
        ''')
        decision_rows = cursor.fetchall()
        decision_dist = {'ALLOW': 0, 'DENY': 0, 'LOCKOUT': 0}
        for decision, cnt in decision_rows:
            decision_dist[decision] = cnt

        # Action distribution (IN/OUT)
        cursor.execute(f'''
            SELECT action, COUNT(*) as cnt
            FROM access_events
            WHERE {where_condition}
            GROUP BY action
        ''')
        action_rows = cursor.fetchall()
        action_dist = {'IN': 0, 'OUT': 0}
        for action, cnt in action_rows:
            action_dist[action] = cnt

        # Score averages
        cursor.execute(f'''
            SELECT AVG(face_score) as avg_face, AVG(voice_score) as avg_voice, AVG(final_score) as avg_final
            FROM access_events
            WHERE {where_condition}
        ''')
        score_avg = cursor.fetchone()
        score_stats = {
            'avg_face': float(score_avg[0] or 0),
            'avg_voice': float(score_avg[1] or 0),
            'avg_final': float(score_avg[2] or 0)
        }

        # Total events count
        cursor.execute(f'''SELECT COUNT(*) FROM access_events WHERE {where_condition}''')
        total_events = cursor.fetchone()[0]

        # Success rate
        cursor.execute(f'''SELECT COUNT(*) FROM access_events WHERE {where_condition} AND genai_decision='ALLOW' ''')
        success_count = cursor.fetchone()[0]
        success_rate = (success_count / total_events * 100) if total_events > 0 else 0

        conn.close()

        return jsonify({
            'owner': owner_filter,
            'heatmap': heatmap,
            'trends': trends,
            'decision_dist': decision_dist,
            'action_dist': action_dist,
            'score_stats': score_stats,
            'total_events': total_events,
            'success_rate': success_rate
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def diagnose_imports():
    """Diagnose import issues"""
    print("\n" + "="*60)
    print("[SEARCH] IMPORT DIAGNOSTICS")
    print("="*60)
    
    # Check Python path
    print(f"\n[PYTHON] Python Path:")
    for path in sys.path:
        print(f"  {path}")
    
    # Check if directories are in path
    print(f"\n[FOLDER] Checking FaceNet in sys.path: {FACENET_DIR in sys.path}")
    print(f"[FOLDER] Checking VoiceRecognition in sys.path: {VOICE_DIR in sys.path}")
    
    # Try importing directly
    print("\n[TEST] Direct import tests:")
    
    # Test FaceNet
    try:
        from FaceNet.face_embedder import get_face_embedding
        print("[OK] face_embedder imports successfully")
    except ImportError as e:
        print(f"[ERROR] face_embedder import failed: {e}")
        # Try with full path
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "face_embedder", 
                os.path.join(FACENET_DIR, "face_embedder.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print("[OK] face_embedder loaded via importlib")
        except Exception as e2:
            print(f"[ERROR] importlib also failed: {e2}")
    
    # Test verify_face
    try:
        from FaceNet.verify_face import verify_face
        print("[OK] verify_face imports successfully")
    except ImportError as e:
        print(f"[ERROR] verify_face import failed: {e}")
    
    # Test speechbrain
    try:
        from speechbrain.inference import SpeakerRecognition
        print("[OK] speechbrain imports successfully")
    except ImportError as e:
        print(f"[ERROR] speechbrain import failed: {e}")
    
    print("="*60 + "\n")


def init_face_module():
    """Initialize face recognition module"""
    try:
        print("[USER] Initializing FaceNet module...")
        
        # Check if FaceNet files exist
        if not os.path.exists(FACENET_DIR):
            print(f"[ERROR] FaceNet directory not found at: {FACENET_DIR}")
            return None
        
        # Try direct import
        try:
            # Import the specific functions
            from FaceNet.face_embedder import get_face_embedding
            from FaceNet.verify_face import verify_face
            
            print("[OK] Successfully imported FaceNet modules")
            
        except ImportError as e:
            print(f"[ERROR] Import error: {e}")
            # Try loading modules dynamically
            try:
                import importlib.util
                
                # Load face_embedder
                embedder_path = os.path.join(FACENET_DIR, "face_embedder.py")
                if not os.path.exists(embedder_path):
                    print(f"[ERROR] face_embedder.py not found at: {embedder_path}")
                    return None
                    
                embedder_spec = importlib.util.spec_from_file_location(
                    "face_embedder", 
                    embedder_path
                )
                face_embedder = importlib.util.module_from_spec(embedder_spec)
                sys.modules["face_embedder"] = face_embedder
                embedder_spec.loader.exec_module(face_embedder)
                
                # Load verify_face
                verify_path = os.path.join(FACENET_DIR, "verify_face.py")
                if not os.path.exists(verify_path):
                    print(f"[ERROR] verify_face.py not found at: {verify_path}")
                    return None
                    
                verify_spec = importlib.util.spec_from_file_location(
                    "verify_face", 
                    verify_path
                )
                verify_face_module = importlib.util.module_from_spec(verify_spec)
                sys.modules["verify_face"] = verify_face_module
                verify_spec.loader.exec_module(verify_face_module)
                
                get_face_embedding = face_embedder.get_face_embedding
                verify_face = verify_face_module.verify_face
                
                print("[OK] Successfully loaded FaceNet modules dynamically")
                
            except Exception as dyn_e:
                print(f"[ERROR] Dynamic loading failed: {dyn_e}")
                return None
        
        # Load face database
        db_path = os.path.join(FACENET_DIR, "embeddings", "face_db.pkl")
        if not os.path.exists(db_path):
            print(f"[WARNING] Face database not found at: {db_path}")
            # Create empty database
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            face_db = {}
            with open(db_path, 'wb') as f:
                pickle.dump(face_db, f)
            print("[NOTE] Created empty face database")
        else:
            try:
                with open(db_path, "rb") as f:
                    face_db = pickle.load(f)
                print(f"[CHART] Loaded face database with {len(face_db)} faces")
            except Exception as db_error:
                print(f"[ERROR] Error loading face database: {db_error}")
                face_db = {}
        
        return {
            'get_face_embedding': get_face_embedding,
            'verify_face': verify_face,
            'face_db': face_db,
            'db_path': db_path
        }
        
    except Exception as e:
        print(f"[ERROR] Face module initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def init_voice_module():
    """Initialize voice recognition module with multi-owner support"""
    try:
        print("[VOICE] Initializing MultiOwner VoiceRecognition module...")
        
        if not os.path.exists(VOICE_DIR):
            print(f"[ERROR] VoiceRecognition directory not found at: {VOICE_DIR}")
            return None
        
        # Load all voice templates (multi-owner support)
        templates_dir = os.path.join(VOICE_DIR, "templates")
        os.makedirs(templates_dir, exist_ok=True)
        
        voice_templates = {}
        template_files = [f for f in os.listdir(templates_dir) if f.endswith("_voice_template.pt")]
        
        if template_files:
            print(f"[FOLDER] Found {len(template_files)} voice template(s)")
            for template_file in sorted(template_files):
                owner_name = template_file.replace("_voice_template.pt", "")
                template_path = os.path.join(templates_dir, template_file)
                try:
                    template = torch.load(template_path)
                    voice_templates[owner_name] = template
                    print(f"   [OK] Loaded: {owner_name} ({template.shape})")
                except Exception as e:
                    print(f"   [ERROR] Failed to load {owner_name}: {e}")
        else:
            print(f"[WARNING]  No voice templates found in {templates_dir}")
            print("   Run: python VoiceRecognition/enroll_my_voice.py")
        
        # Try to load speechbrain
        try:
            from speechbrain.inference import SpeakerRecognition
            
            # Load the verifier model
            print("[LOAD] Loading SpeechBrain model...")
            model_dir = os.path.join(VOICE_DIR, "pretrained_models", "spkrec-ecapa-voxceleb")
            os.makedirs(model_dir, exist_ok=True)
            
            verifier = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=model_dir
            )
            
            print("[OK] Voice module loaded with SpeechBrain")
            print(f"👥 Multi-Owner support: {len(voice_templates)} owner(s) registered")
            
            return {
                'verifier': verifier,
                'voice_templates': voice_templates,
                'simulated': False
            }
            
        except ImportError as sb_error:
            print(f"[WARNING] SpeechBrain not available: {sb_error}")
            print("[CONFIG] Creating simulated voice verifier...")
            
            # Create a simulated verifier
            class SimulatedVerifier:
                def encode_batch(self, waveform):
                    # Return random embedding
                    return torch.randn(1, 1, 192)
                    
                def __str__(self):
                    return "Simulated Voice Verifier"
            
            return {
                'verifier': SimulatedVerifier(),
                'voice_templates': voice_templates,
                'simulated': True
            }
        except Exception as model_error:
            print(f"[ERROR] SpeechBrain model loading failed: {model_error}")
            print("[CONFIG] Falling back to simulated verifier...")
            
            class SimulatedVerifier:
                def encode_batch(self, waveform):
                    return torch.randn(1, 1, 192)
                    
            return {
                'verifier': SimulatedVerifier(),
                'voice_templates': voice_templates,
                'simulated': True
            }
            
    except Exception as e:
        print(f"[ERROR] Voice module initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_face_score(face_module, image_path):
    """Get face verification score with actual model verification
    Returns: (score, owner) tuple"""
    if face_module is None:
        print("[WARNING]  Face module not initialized - returning simulated score")
        return 0.0, "unknown"
    
    try:
        # Get embedding from the live image
        live_emb = face_module['get_face_embedding'](image_path)
        if live_emb is None:
            print("[ERROR] No face detected in image")
            return 0.0, "unknown"
        
        # Verify against stored embeddings
        owner, score, allowed = face_module['verify_face'](
            live_emb,
            face_module['face_db'],
            FACE_THRESHOLD
        )
        
        print(f"[USER] Face verification result: owner={owner}, score={score:.3f}, allowed={allowed}")
        
        if owner and owner != "Unknown":
            print(f"[USER] Face recognized as: {owner} (score: {score:.3f})")
            # Normalize score from [-1, 1] to [0, 1]
            normalized_score = max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
            return normalized_score, owner.lower()
        else:
            print(f"[ERROR] Face not recognized (score: {score:.3f})")
            return 0.0, "unknown"
    except Exception as e:
        print(f"[ERROR] Face verification error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, "unknown"

def get_voice_score(voice_module, audio_path):
    """Get voice verification score with multi-owner support"""
    if voice_module is None or not voice_module.get('voice_templates'):
        print("[WARNING]  Voice module not initialized - returning 0.0")
        return 0.0, "Unknown"
    
    try:
        print(f"[VOICE] Processing audio file: {audio_path}")
        print(f"[VOICE] File size: {os.path.getsize(audio_path)} bytes")
        
        # Load audio using pydub with format detection
        waveform = None
        try:
            # Try to detect format from file extension
            file_ext = os.path.splitext(audio_path)[1].lower()
            print(f"[VOICE] Detected format: {file_ext}")
            
            # Try loading with explicit format for WebM
            if file_ext in ['.webm', '.ogg']:
                try:
                    audio = AudioSegment.from_file(audio_path, format='webm')
                except:
                    audio = AudioSegment.from_file(audio_path, format='ogg')
            else:
                audio = AudioSegment.from_file(audio_path)
            
            # Standardize to 16kHz, Mono, 16-bit
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # Convert to numpy array and normalize to [-1, 1]
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2**15)  # Normalize 16-bit PCM
            
            # Ensure minimum audio length
            if len(samples) < 8000:  # Less than 0.5 seconds at 16kHz
                print(f"[WARNING]  Audio too short: {len(samples)} samples")
                return 0.0, "Unknown"
            
            waveform = torch.from_numpy(samples).unsqueeze(0)
            
            print(f"[OK] Audio loaded: {len(samples)} samples ({len(samples)/16000:.2f}s)")
            
        except Exception as audio_err:
            print(f"[WARNING]  pydub failed: {audio_err}")
            # Try librosa as fallback
            try:
                import librosa
                samples, sr = librosa.load(audio_path, sr=16000, mono=True)
                print(f"[OK] Audio loaded with librosa: {len(samples)} samples")
                
                if len(samples) < 8000:
                    print(f"[WARNING]  Audio too short: {len(samples)} samples")
                    return 0.0, "Unknown"
                    
                waveform = torch.from_numpy(samples).float().unsqueeze(0)
            except Exception as lib_err:
                print(f"[ERROR] Both pydub and librosa failed: {lib_err}")
                import traceback
                traceback.print_exc()
                return 0.0, "Unknown"
        
        if waveform is None:
            print("[ERROR] Failed to load audio")
            return 0.0, "Unknown"
        
        # Get embedding from live audio
        verifier = voice_module['verifier']
        emb_live = verifier.encode_batch(waveform)
        
        # Handle tensor dimensions
        if emb_live.dim() == 3:
            emb_live = emb_live.mean(dim=1)
        elif emb_live.dim() == 1:
            emb_live = emb_live.unsqueeze(0)
        
        # Compare against ALL owner templates (multi-owner support)
        voice_templates = voice_module['voice_templates']
        scores = {}
        
        for owner_name, emb_template in voice_templates.items():
            if isinstance(emb_template, np.ndarray):
                emb_template = torch.from_numpy(emb_template).float()
            
            if emb_template.dim() == 1:
                emb_template = emb_template.unsqueeze(0)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                emb_template.view(1, -1),
                emb_live.view(1, -1),
                dim=1
            )
            scores[owner_name] = float(similarity[0].item())
        
        # Find best match
        best_owner = max(scores, key=scores.get) if scores else "Unknown"
        best_score = scores[best_owner] if best_owner in scores else 0.0
        
        print(f"\n[SEARCH] Voice Scores:")
        for owner, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {owner}: {score:.4f}")
        
        # Normalize score from [-1, 1] to [0, 1]
        normalized_score = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
        
        print(f"[VOICE] Best match: {best_owner} (raw: {best_score:.4f}, normalized: {normalized_score:.4f})")
        
        return normalized_score, best_owner
    except Exception as e:
        print(f"[ERROR] Voice verification error: {e}")
        import traceback
        traceback.print_exc()
        if emb_template.dim() == 1:
            emb_template = emb_template.unsqueeze(0)
        
        # Calculate cosine similarity
        score = torch.nn.functional.cosine_similarity(
            emb_template.view(1, -1), 
            emb_live.view(1, -1), 
            dim=1
        )
        score_val = float(score[0].item())
        
        print(f"[VOICE] Voice verification raw score: {score_val:.3f}")
        
        # Normalize score from [-1, 1] to [0, 1]
        normalized_score = max(0.0, min(1.0, (score_val + 1.0) / 2.0))
        print(f"[VOICE] Voice verification normalized score: {normalized_score:.3f}")
        
        return normalized_score
    except Exception as e:
        print(f"[ERROR] Voice verification error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def process_attempt_background(pin, audio_path, img_path, q, tmp_dir):
    """Background processing function (from old code)"""
    try:
        q.put({"status": "progress", "message": "Checking Security PIN..."})
        if not PINSecurityManager.verify_pin(pin, SYSTEM_PIN_HASH):
            q.put({"status": "done", "result": "DENIED", "reason": "Invalid PIN"})
            return

        q.put({"status": "progress", "message": "Processing Voice Biometrics..."})
        
        # Process voice
        v_score = get_voice_score(voice_module, audio_path)
        q.put({"status": "progress", "message": f"Voice Score: {v_score:.3f}"})

        q.put({"status": "progress", "message": "Comparing Facial Features..."})
        f_score = get_face_score(face_module, img_path)
        q.put({"status": "progress", "message": f"Face Score: {f_score:.3f}"})

        # Normalize [-1, 1] to [0, 1]
        def norm(s): 
            return max(0.0, min(1.0, (float(s) + 1.0) / 2.0))
        
        v_norm = norm(v_score)
        f_norm = norm(f_score)
        combined = (VOICE_WEIGHT * v_norm) + (FACE_WEIGHT * f_norm)
        
        details = {
            "voice": round(v_score, 3), 
            "face": round(f_score, 3), 
            "combined": round(combined, 3)
        }

        print(f"[LOCK] Combined Score: {combined:.3f} (Threshold: {COMBINED_THRESHOLD})")

        if combined >= COMBINED_THRESHOLD:
            q.put({"status": "done", "result": "GRANTED", "details": details})
        else:
            q.put({"status": "done", "result": "DENIED", "reason": "Biometric score too low", "details": details})

    except Exception as e:
        print(f"[ERROR] Critical process error: {e}")
        q.put({"status": "done", "result": "DENIED", "reason": f"System Error: {str(e)}"})
    finally:
        q.put(None)

@app.route('/')
def index():
    """Render main web interface"""
    return render_template('index.html')

@app.route('/password-change')
def password_change():
    """Render standalone password change page"""
    return render_template('password_change.html')

@app.route('/reset-lockout')
def reset_lockout_page():
    """Render standalone lockout reset page"""
    return render_template('reset_lockout.html')

@app.route('/api/status')
def system_status():
    """Get system status information"""
    try:
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        
        # Get counts
        cursor.execute("SELECT COUNT(*) FROM access_events")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'ALLOW'")
        allowed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE genai_decision = 'DENY'")
        denied = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM access_events WHERE lockout_active = 1")
        lockouts = cursor.fetchone()[0]
        
        # Check current lockout
        cursor.execute("""
            SELECT lockout_active FROM access_events 
            WHERE lockout_active = 1 
            ORDER BY record_created_at DESC LIMIT 1
        """)
        lockout_result = cursor.fetchone()
        is_locked = lockout_result[0] if lockout_result else False
        
        conn.close()
        
        # Calculate success rate
        success_rate = allowed / total if total > 0 else 0
        
        return jsonify({
            'status': 'online',
            'total_events': total,
            'allowed': allowed,
            'denied': denied,
            'success_rate': success_rate,
            'lockouts': lockouts,
            'lockout_active': bool(is_locked),
            'face_module': face_module is not None,
            'voice_module': voice_module is not None,
            'ai_module': ai_analyzer.api_key is not None,
            'system_mode': get_system_mode(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate():
    """Main authentication endpoint with background processing option"""
    try:
        print("\n" + "="*50)
        print("[LOCK] NEW AUTHENTICATION ATTEMPT")
        print("="*50)

        # Block authentication when in vacation mode
        current_mode = get_system_mode()
        if current_mode == 'vacation':
            return jsonify({
                'success': False,
                'decision': 'DENY',
                'risk_level': 'HIGH',
                'explanation': 'System is in Vacation Mode. Access temporarily disabled.',
                'scores': {},
                'lockout': False,
                'failed_attempts': None,
                'event_id': None,
                'timestamp': datetime.now().isoformat()
            }), 403

        # --- FULL DEBUG LOGGING ---
        print("[INFO] RAW request.form:", dict(request.form))
        print("[INFO] RAW request.files:", list(request.files.keys()))

        # Get form data
        data = request.form
        # Accept IN/OUT or legacy LOCK/UNLOCK and normalize to IN/OUT
        raw_action = data.get('action', 'IN')
        action = raw_action.strip().upper() if isinstance(raw_action, str) else 'IN'
        if action in ('UNLOCK', 'IN', 'ENTER', 'ENTERING'):
            action = 'IN'
        elif action in ('LOCK', 'OUT', 'EXIT', 'EXITING'):
            action = 'OUT'
        else:
            action = 'IN'
        manual_date = data.get('manual_date')
        manual_time = data.get('manual_time')
        entered_pin = data.get('pin', '')

        from datetime import datetime as dt
        if not manual_date:
            manual_date = dt.now().strftime("%Y-%m-%d")
            print(f"[WARNING] manual_date not provided, using today: {manual_date}")
        if not manual_time:
            manual_time = dt.now().strftime("%H:%M")
            print(f"[WARNING] manual_time not provided, using current time: {manual_time}")

        # Always create manual_datetime
        manual_datetime = f"{manual_date} {manual_time}:00"
        # Robustly compute day_of_week from manual_date
        day_of_week = None
        date_formats = ["%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y"]
        for fmt in date_formats:
            try:
                day_of_week = dt.strptime(manual_date, fmt).strftime("%A")
                break
            except Exception:
                continue
        if not day_of_week:
            print(f"[ERROR] Could not parse manual_date '{manual_date}' for day_of_week. Supported formats: {date_formats}")
            return jsonify({'success': False, 'error': f'Invalid date format: {manual_date}. Please use YYYY-MM-DD or DD-MMM-YYYY.'}), 400

        # --- FORCED DEBUG LOGGING ---
        print(f"[FORCE-DEBUG] manual_date received: {manual_date}")
        print(f"[FORCE-DEBUG] manual_time received: {manual_time}")
        print(f"[FORCE-DEBUG] manual_datetime constructed: {manual_datetime}")
        print(f"[FORCE-DEBUG] day_of_week computed: {day_of_week}")
        print(f"[INFO] Action: {action}")


        # Verify PIN using hash comparison
        pin_valid = PINSecurityManager.verify_pin(entered_pin, SYSTEM_PIN_HASH)
        print(f"PIN Valid: {pin_valid}")

        # Robust input parsing and module initialization
        if 'face_image' not in request.files or 'voice_audio' not in request.files:
            print("[ERROR] Missing biometric files")
            return jsonify({
                'success': False,
                'error': 'Please provide both face image and voice recording',
                'decision': 'ERROR',
                'scores': {'pin': pin_valid, 'face': 0, 'voice': 0, 'behavior': 0, 'final': 0}
            }), 400

        face_file = request.files['face_image']
        voice_file = request.files['voice_audio']

        # Debug: log file size and type
        face_file.seek(0, os.SEEK_END)
        face_size = face_file.tell()
        face_file.seek(0)
        voice_file.seek(0, os.SEEK_END)
        voice_size = voice_file.tell()
        voice_file.seek(0)
        print(f"[DEBUG] Received face_image: filename={face_file.filename}, content_type={face_file.content_type}, size={face_size}")
        print(f"[DEBUG] Received voice_audio: filename={voice_file.filename}, content_type={voice_file.content_type}, size={voice_size}")

        if face_file.filename == '' or voice_file.filename == '' or face_size == 0 or voice_size == 0:
            print("[ERROR] Empty files submitted or zero size")
            return jsonify({
                'success': False,
                'error': 'No file selected or file is empty',
                'decision': 'ERROR',
                'scores': {'pin': pin_valid, 'face': 0, 'voice': 0, 'behavior': 0, 'final': 0}
            }), 400

        try:
            print(f"Face file: {face_file.filename}, {face_file.content_type}, size: {face_size}")
            print(f"[VOICE] Voice file: {voice_file.filename}, {voice_file.content_type}, size: {voice_size}")

            tmp_dir = tempfile.mkdtemp()
            temp_files = []
            face_path = os.path.join(tmp_dir, f'face_{uuid.uuid4()}.jpg')
            voice_path = os.path.join(tmp_dir, f'voice_{uuid.uuid4()}.webm')

            face_file.save(face_path)
            voice_file.save(voice_path)
            temp_files.extend([face_path, voice_path])

            print(f"[BACKUP] Files saved to: {tmp_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to save files: {e}")
            return jsonify({
                'success': False,
                'error': f'File save error: {str(e)}',
                'decision': 'ERROR',
                'scores': {'pin': pin_valid, 'face': 0, 'voice': 0, 'behavior': 0, 'final': 0}
            }), 500

        # Process face (returns score and owner)
        try:
            face_score, face_owner = get_face_score(face_module, face_path)
        except Exception as e:
            print(f"[ERROR] Face scoring failed: {e}")
            face_score, face_owner = 0, 'unknown'
        print(f"Face Score: {face_score:.3f}, Owner: {face_owner}")

        # Process voice (returns score and owner name)
        try:
            voice_result = get_voice_score(voice_module, voice_path)
            if isinstance(voice_result, tuple):
                voice_score, voice_owner = voice_result
            else:
                voice_score = voice_result
                voice_owner = "unknown"
        except Exception as e:
            print(f"[ERROR] Voice scoring failed: {e}")
            voice_score, voice_owner = 0, 'unknown'
        print(f"[VOICE] Voice Score: {voice_score:.3f}, Owner: {voice_owner}")
        
        # [OK] CROSS-VERIFICATION: Check if voice owner matches face owner
        owner_mismatch = False
        if face_owner != "unknown" and voice_owner != "unknown":
            if face_owner.lower() != voice_owner.lower():
                owner_mismatch = True
                print(f"[WARNING]  OWNER MISMATCH DETECTED!")
                print(f"   Face Owner: {face_owner}")
                print(f"   Voice Owner: {voice_owner}")
        
        # Determine actual owner (prefer face if both present, otherwise voice)
        # Force lowercase to match DB constraint (rithika|sid|unknown)
        determined_owner = (face_owner if face_owner != "unknown" else voice_owner if voice_owner != "unknown" else "unknown").lower()
        print(f"[OK] Determined Owner: {determined_owner}")
        
        # Calculate behavior score (now also reflects biometric quality)
        behavior_score = calculate_behavior_score(
            manual_datetime, action, face_score=face_score, voice_score=voice_score, pin_valid=pin_valid
        )
        

        # Calculate final score (weighted average)
        weights = {'pin': 0.3, 'face': 0.35, 'voice': 0.35}
        final_score = (
            weights['pin'] * (1.0 if pin_valid else 0.0) +
            weights['face'] * face_score +
            weights['voice'] * voice_score
        )

        # Penalize if owner mismatch detected
        if owner_mismatch:
            final_score *= 0.6  # 40% penalty for owner mismatch
            print(f"Final Score after mismatch penalty: {final_score:.3f}")

        print(f"Behavior Score: {behavior_score:.3f}")
        print(f"Final Score: {final_score:.3f}")

        # --- STRICT BIOMETRIC THRESHOLD ---

        # Ensure failed_attempts is initialized before use
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT failed_attempt_count FROM access_events 
            ORDER BY record_created_at DESC LIMIT 1
        """)
        last_result = cursor.fetchone()
        failed_attempts = last_result[0] if last_result else 0
        cursor.close()
        conn.close()

        # --- STRICT FINAL SCORE THRESHOLD ---
        if final_score < 0.85:
            failed_attempts += 1

            # Store investigation image for alerts
            recent_images = store_investigation_image(face_path)

            # Enforce lockout after 5 consecutive denials
            lockout_active = failed_attempts >= 5
            decision = 'LOCKOUT' if lockout_active else 'DENY'
            explanation = (
                'Multiple consecutive failures detected. System locked for security.'
                if lockout_active else
                'Combined biometric score too low. Access denied.'
            )

            # Run model + GenAI analysis even in low-score path (decision still enforced below)
            model_explanation = None
            try:
                from Model.explanation_model import LightweightExplanationModel
                db_path = os.path.abspath("smart_lock_events.db")
                model_dir = os.path.abspath("Model")
                if determined_owner not in (None, '', 'unknown'):
                    model = LightweightExplanationModel(db_path, model_dir, determined_owner)
                    time_of_day = manual_datetime.split()[1] if manual_datetime and len(manual_datetime.split()) > 1 else "00:00"
                    model_context = {
                        "owner": determined_owner,
                        "day_of_week": day_of_week,
                        "time": time_of_day,
                        "action": action,
                        "pin_valid": int(pin_valid),
                        "face_score": face_score,
                        "voice_score": voice_score,
                        "behavior_score": behavior_score,
                        "final_score": final_score,
                        "genai_decision": None,
                        "genai_risk_level": None
                    }
                    model_explanation = model.predict(model_context)
            except Exception as model_err:
                print(f"[MODEL] Low-score path model prediction failed: {model_err}")

            ai_result = {'decision': decision, 'risk_level': 'HIGH', 'explanation': explanation}
            try:
                attempt_data = {
                    'action': action,
                    'manual_datetime': manual_datetime,
                    'day_of_week': day_of_week,
                    'pin_valid': pin_valid,
                    'face_score': face_score,
                    'voice_score': voice_score,
                    'behavior_score': behavior_score,
                    'final_score': final_score,
                    'failed_attempt_count': failed_attempts,
                    'lockout_active': lockout_active,
                    'owner': determined_owner,
                    'face_owner': face_owner,
                    'voice_owner': voice_owner,
                    'owner_mismatch': owner_mismatch,
                    'model_explanation': model_explanation
                }
                ai_analysis = ai_analyzer.analyze_attempt(attempt_data)
                if ai_analysis and ai_analysis.get('explanation'):
                    ai_result['explanation'] = f"{explanation} | {ai_analysis['explanation']}"
            except Exception as ai_err:
                print(f"[AI] Low-score path GenAI analysis failed: {ai_err}")

            # Send alerts on 2-3 failed attempts (with images)
            if failed_attempts in [2, 3]:
                print(f"[WARNING]  ALERT: {failed_attempts} failed attempts detected - Sending SMS alert...")
                try:
                    send_sms(image_paths=recent_images if recent_images else None)
                except Exception as e:
                    print(f"[ERROR] Error sending SMS alert: {e}")

            # Trigger emergency call on lockout
            if lockout_active:
                print(f"ALERT: SYSTEM LOCKED - Sending emergency alert...")
                try:
                    make_call()
                except Exception as e:
                    print(f"[ERROR] Error sending emergency alert: {e}")

            event_id = db_logger.log_event(
                action=action,
                day_of_week=day_of_week,
                entry_time=manual_datetime if action == 'IN' else None,
                exit_time=manual_datetime if action == 'OUT' else None,
                event_datetime=manual_datetime,
                pin_valid=pin_valid,
                face_score=face_score,
                voice_score=voice_score,
                behavior_score=behavior_score,
                final_score=final_score,
                genai_decision=ai_result['decision'],
                genai_risk_level=ai_result['risk_level'],
                genai_explanation=ai_result['explanation'],
                failed_attempt_count=failed_attempts,
                lockout_active=lockout_active,
                owner=determined_owner,
                person_status='OUTSIDE',
                face_owner=face_owner,
                voice_owner=voice_owner,
                model_explanation=model_explanation,
                owner_feedback=None,
                genai_updated_explanation=None,
                alert_sent=failed_attempts in [2, 3],
                alert_type='telegram'
            )
            try:
                from Alert_system.sms import send_telegram
                send_telegram(f"ALERT: Low combined biometric score for {determined_owner} at {manual_datetime}.")
            except Exception as e:
                print(f"[ALERT] Telegram alert failed: {e}")
            return jsonify({
                'success': False,
                'decision': ai_result['decision'],
                'risk_level': ai_result['risk_level'],
                'explanation': ai_result['explanation'],
                'scores': {
                    'pin': pin_valid,
                    'face': round(face_score, 3),
                    'voice': round(voice_score, 3),
                    'behavior': round(behavior_score, 3),
                    'final': round(final_score, 3)
                },
                'lockout': lockout_active,
                'failed_attempts': failed_attempts,
                'event_id': event_id,
                'timestamp': datetime.now().isoformat()
            })

        # If we reach here, final_score >= 0.85, so proceed to decision tree and GenAI
        # ...existing code for decision tree, GenAI, owner pattern match, mild alert, feedback, retraining...

        
        # Get current failure streak
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute("""
            SELECT failed_attempt_count FROM access_events 
            ORDER BY record_created_at DESC LIMIT 1
        """)
        last_result = cursor.fetchone()
        failed_attempts = last_result[0] if last_result else 0
        
        print(f"[CHART] Failed Attempts Streak: {failed_attempts}")
        


        # --- INTEGRATE OWNER-SPECIFIC DECISION TREE MODEL (BOTH OWNERS) ---
        model_explanation = None
        try:
            from Model.explanation_model import LightweightExplanationModel
            db_path = os.path.abspath("smart_lock_events.db")
            model_dir = os.path.abspath("Model")
            if determined_owner not in (None, '', 'unknown'):
                model = LightweightExplanationModel(db_path, model_dir, determined_owner)
                time_of_day = manual_datetime.split()[1] if manual_datetime and len(manual_datetime.split()) > 1 else "00:00"
                print(f"[DEBUG] manual_datetime passed to model: {manual_datetime}")
                print(f"[DEBUG] day_of_week passed to model: {day_of_week}")
                model_context = {
                    "owner": determined_owner,
                    "day_of_week": day_of_week,
                    "time": time_of_day,
                    "action": action,
                    "pin_valid": int(pin_valid),
                    "face_score": face_score,
                    "voice_score": voice_score,
                    "behavior_score": behavior_score,
                    "final_score": final_score,
                    "genai_decision": None,
                    "genai_risk_level": None
                }
                model_explanation = model.predict(model_context)
                print(f"[MODEL] Owner-specific model explanation: {model_explanation}")
        except Exception as e:
            print(f"[MODEL] Model prediction failed: {e}")

        # Prepare data for GenAI analysis, including model prediction
        attempt_data = {
            'action': action,
            'manual_datetime': manual_datetime,
            'day_of_week': day_of_week,
            'time_of_day': time_of_day if 'time_of_day' in locals() else None,
            'pin_valid': pin_valid,
            'face_score': face_score,
            'voice_score': voice_score,
            'behavior_score': behavior_score,
            'final_score': final_score,
            'failed_attempt_count': failed_attempts,
            'lockout_active': False,
            'owner': determined_owner,
            'face_owner': face_owner,
            'voice_owner': voice_owner,
            'owner_mismatch': owner_mismatch,
            'model_explanation': model_explanation
        }

        # Consult GenAI for decision, now with model_explanation
        print("\n[AI] CONSULTING GENAI (with model explanation)...")
        # Pass model_explanation as explicit context to GenAI
        attempt_data['model_explanation'] = model_explanation
        ai_result = ai_analyzer.analyze_attempt(attempt_data)

        # Ensure GenAI explanation references model_explanation if present
        if model_explanation:
            # If model_explanation is a dict, extract explanation string and decision
            if isinstance(model_explanation, dict):
                model_expl_str = model_explanation.get('explanation', str(model_explanation))
                model_decision = model_explanation.get('decision', None)
                trigger_alert = model_explanation.get('trigger_alert', False)
            else:
                model_expl_str = str(model_explanation)
                model_decision = None
                trigger_alert = False
            # Keep model-driven decisioning, but always preserve both Model + GenAI explanations.
            if model_decision == 'ALLOW':
                ai_result['decision'] = 'ALLOW'
                if 'risk_level' in model_explanation:
                    ai_result['risk_level'] = model_explanation['risk_level']

            genai_expl = str(ai_result.get('explanation', '')).strip()
            if model_expl_str and genai_expl:
                ai_result['explanation'] = f"Model: {model_expl_str} | GenAI: {genai_expl}"
            elif model_expl_str:
                ai_result['explanation'] = f"Model: {model_expl_str}"
            elif genai_expl:
                ai_result['explanation'] = f"GenAI: {genai_expl}"
            # --- NEW: If trigger_alert is set, always send Telegram and store feedback ---
            if trigger_alert:
                try:
                    from Alert_system.sms import send_telegram
                    send_telegram(f"Hey {determined_owner.title()}, late entry detected: {manual_datetime}. Reason: {model_expl_str}. Please reply with reason or confirmation.")
                except Exception as e:
                    print(f"[ALERT] Telegram alert failed: {e}")
                # Placeholder: Simulate user feedback (to be replaced by Telegram webhook integration)
                owner_feedback = "User confirmed late entry."
                genai_updated_explanation = f"Late entry confirmed by user: {owner_feedback}"

        print(f"[OK] GenAI Decision: {ai_result['decision']}")
        print(f"[WARNING]  Risk Level: {ai_result['risk_level']}")
        print(f"[IDEA] Explanation: {ai_result['explanation']}")

        # If unusual but authorized (ALLOW, medium/high risk, unusual time), send Telegram and await feedback
        alert_sent = False
        alert_type = None
        if ai_result['decision'] == 'ALLOW' and ai_result['risk_level'] in ['MEDIUM', 'HIGH']:
            try:
                from Alert_system.sms import send_telegram
                send_telegram(f"Hey {determined_owner.title()}, unusual access detected: {manual_datetime}. Reason: {ai_result['explanation']}")
                alert_sent = True
                alert_type = 'telegram'
            except Exception as e:
                print(f"[ALERT] Telegram alert failed: {e}")

        # Owner feedback loop (pseudo, to be implemented with Telegram bot webhook)
        owner_feedback = None
        genai_updated_explanation = None
        # If late/exception entry, send Telegram and await feedback
        is_late_entry = False
        model_expl_str = None
        if model_explanation:
            if isinstance(model_explanation, dict):
                model_expl_str = model_explanation.get('explanation', str(model_explanation))
            else:
                model_expl_str = str(model_explanation)
        if model_expl_str and ("late" in model_expl_str.lower() or "exception" in model_expl_str.lower()):
            is_late_entry = True
            try:
                from Alert_system.sms import send_telegram
                send_telegram(f"Late/exception entry detected for {determined_owner} at {manual_datetime}. Reason: {model_explanation}. Please reply with reason or confirmation.")
                alert_sent = True
                alert_type = 'telegram_late_entry'
            except Exception as e:
                print(f"[ALERT] Telegram alert failed: {e}")
            # Placeholder: Simulate user feedback (to be replaced by Telegram webhook integration)
            owner_feedback = "User confirmed late entry."
            genai_updated_explanation = f"Late entry confirmed by user: {owner_feedback}"
            # TODO: Save feedback and retrain decision tree here

        
        # Update failure count
        if ai_result['decision'] == 'DENY':
            failed_attempts += 1
        elif ai_result['decision'] == 'ALLOW':
            failed_attempts = 0
        
        # STORE CAPTURED IMAGE FOR INVESTIGATION
        recent_images = store_investigation_image(face_path)
        
        # Check if lockout should be activated (5 consecutive denials)
        lockout_active = (ai_result['decision'] == 'LOCKOUT')
        if failed_attempts >= 5:
            lockout_active = True
            ai_result['decision'] = 'LOCKOUT'
            ai_result['risk_level'] = 'HIGH'
            ai_result['explanation'] = 'Multiple consecutive failures detected. System locked for security.'
        
        # ALERT SYSTEM INTEGRATION
        # Trigger SMS alert on 2-3 denials WITH captured images
        if failed_attempts in [2, 3]:
            print(f"[WARNING]  ALERT: {failed_attempts} failed attempts detected - Sending SMS alert...")
            print(f"Sending {len(captured_images)} captured image(s) for investigation...")

            try:
                send_sms(image_paths=recent_images if recent_images else None)
            except Exception as e:
                print(f"[ERROR] Error sending SMS alert: {e}")
        
        # Trigger emergency call on system lockout
        if ai_result['decision'] == 'LOCKOUT':
            print(f"ALERT: SYSTEM LOCKED - Sending emergency alert...")
            try:
                make_call()
            except Exception as e:
                print(f"[ERROR] Error sending emergency alert: {e}")
        
        # Get current door status
        conn2 = sqlite3.connect('smart_lock_events.db')
        cursor2 = conn2.cursor()
        cursor2.execute("""
            SELECT person_status FROM access_events 
            ORDER BY record_created_at DESC LIMIT 1
        """)
        last_person = cursor2.fetchone()
        current_person_status = last_person[0] if last_person else 'OUTSIDE'
        conn2.close()
        
        # Update person status based on action and decision
        if ai_result['decision'] == 'ALLOW':
            new_person_status = 'INSIDE' if action == 'IN' else 'OUTSIDE'
        else:
            new_person_status = current_person_status
        
        # Log the event with new columns
        event_id = db_logger.log_event(
            action=action,
            day_of_week=day_of_week,
            entry_time=manual_datetime if action == 'IN' else None,
            exit_time=manual_datetime if action == 'OUT' else None,
            event_datetime=manual_datetime,
            pin_valid=pin_valid,
            face_score=face_score,
            voice_score=voice_score,
            behavior_score=behavior_score,
            final_score=final_score,
            genai_decision=ai_result['decision'],
            genai_risk_level=ai_result['risk_level'],
            genai_explanation=ai_result['explanation'],
            failed_attempt_count=failed_attempts,
            lockout_active=lockout_active,
            owner=determined_owner,
            person_status=new_person_status,
            face_owner=face_owner,
            voice_owner=voice_owner,
            model_explanation=model_explanation,
            owner_feedback=owner_feedback,
            genai_updated_explanation=genai_updated_explanation,
            alert_sent=alert_sent,
            alert_type=alert_type
        )
        
        print(f"[NOTE] Logged as Event ID: {event_id}")
        
        conn.close()
        
        # Clean up temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass
        
        # Remove temp directory
        try:
            os.rmdir(tmp_dir)
        except:
            pass
        
        print("="*50 + "\n")
        
        return jsonify({
            'success': ai_result['decision'] == 'ALLOW',
            'decision': ai_result['decision'],
            'risk_level': ai_result['risk_level'],
            'explanation': ai_result['explanation'],
            'scores': {
                'pin': pin_valid,
                'face': round(face_score, 3),
                'voice': round(voice_score, 3),
                'behavior': round(behavior_score, 3),
                'final': round(final_score, 3)
            },
            'lockout': lockout_active,
            'failed_attempts': failed_attempts,
            'event_id': event_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Authentication error: {e}\n{tb}")
        fallback_explanation = "No explanation available. Please check system logs."
        return jsonify({
            'success': False,
            'error': str(e),
            'decision': 'ERROR',
            'risk_level': 'UNKNOWN',
            'explanation': fallback_explanation,
            'scores': {},
            'lockout': False,
            'failed_attempts': None,
            'event_id': None,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/events')
def get_events():
    """Get recent access events"""
    try:
        limit = request.args.get('limit', 50, type=int)
        events = db_logger.get_recent_events(limit)
        
        # Format for frontend
        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event['id'],
                'action': event['action'],
                'time': event.get('event_datetime') or event.get('record_created_at'),
                'decision': event['genai_decision'],
                'risk_level': event['genai_risk_level'],
                'explanation': event['genai_explanation'],
                'pin_valid': bool(event['pin_valid']),
                'face_score': event['face_score'],
                'voice_score': event['voice_score'],
                'final_score': event['final_score'],
                'failed_attempts': event['failed_attempt_count'],
                'lockout': bool(event['lockout_active'])
            })
        
        return jsonify({'events': formatted_events})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Request OTP for lockout reset
@app.route('/api/lockout/request-otp', methods=['POST'])
def request_lockout_otp():
    try:
        email = RECEIVER_EMAIL.lower()
        otp = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.now() + timedelta(minutes=10)
        try:
            conn = sqlite3.connect('smart_lock_events.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO lockout_otps (email, otp, expires_at, verified)
                VALUES (?, ?, ?, 0)
                ON CONFLICT(email) DO UPDATE SET
                    otp=excluded.otp,
                    expires_at=excluded.expires_at,
                    verified=0
            ''', (email, otp, expires_at.isoformat()))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"[ERROR] Failed to store lockout OTP: {db_err}")
            return jsonify({'success': False, 'message': 'Failed to store OTP'}), 500
        subject = "SmartLock Lockout Reset - OTP Verification"
        body = f"""Dear User,\n\nYou have requested to reset the lockout state.\n\nYour One-Time Password (OTP) is: {otp}\n\nThis OTP is valid for 10 minutes only.\n\nIf you did not request this, please ignore this email.\n\nSecurity Notice: Never share this OTP with anyone.\n\nBest regards,\nSmartLock System"""
        try:
            send_email(subject, body, to_email=email)
            print(f"[OK] Lockout OTP sent to {email}")
            return jsonify({'success': True, 'message': f'OTP sent to {email}', 'email': email}), 200
        except Exception as email_err:
            print(f"[ERROR] Failed to send lockout OTP email: {email_err}")
            return jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500
    except Exception as e:
        print(f"[ERROR] Lockout OTP request error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Verify OTP for lockout reset
@app.route('/api/lockout/verify-otp', methods=['POST'])
def verify_lockout_otp():
    try:
        email = RECEIVER_EMAIL.lower()
        data = request.get_json()
        otp_input = data.get('otp', '').strip()
        if not otp_input:
            return jsonify({'success': False, 'message': 'OTP is required'}), 400
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('SELECT otp, expires_at, verified FROM lockout_otps WHERE email=?', (email,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return jsonify({'success': False, 'message': 'No OTP request found. Please request OTP first.'}), 400
        stored_otp, expires_at_str, verified = row
        expires_at = datetime.fromisoformat(expires_at_str)
        if datetime.now() > expires_at:
            cursor.execute('DELETE FROM lockout_otps WHERE email=?', (email,))
            conn.commit()
            conn.close()
            return jsonify({'success': False, 'message': 'OTP has expired. Please request a new one.'}), 400
        if stored_otp != otp_input:
            conn.close()
            return jsonify({'success': False, 'message': 'Incorrect OTP. Please try again.'}), 400
        cursor.execute('UPDATE lockout_otps SET verified=1 WHERE email=?', (email,))
        conn.commit()
        conn.close()
        print(f"[OK] Lockout OTP verified for {email}")
        return jsonify({'success': True, 'message': 'OTP verified successfully', 'email': email}), 200
    except Exception as e:
        print(f"[ERROR] Lockout OTP verification error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Perform lockout reset after OTP verification
@app.route('/api/lockout/reset', methods=['POST'])
def reset_lockout_otp():
    try:
        email = RECEIVER_EMAIL.lower()
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('SELECT verified FROM lockout_otps WHERE email=?', (email,))
        row = cursor.fetchone()
        if not row or not row[0]:
            conn.close()
            return jsonify({'success': False, 'message': 'OTP verification required. Please verify OTP first.'}), 403
        # Reset lockout using the same logic as before
        success = db_logger.reset_lockout()
        cursor.execute('DELETE FROM lockout_otps WHERE email=?', (email,))
        conn.commit()
        conn.close()
        if success:
            print("[RETRY] Lockout reset by OTP")
            return jsonify({'success': True, 'message': 'Lockout state reset successfully'})
        else:
            return jsonify({'success': False, 'message': 'No active lockout found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# --- System Mode (Vacation / In Town) ---
@app.route('/api/mode', methods=['GET'])
def get_mode():
    return jsonify({'mode': get_system_mode()})

@app.route('/api/mode/vacation', methods=['POST'])
def set_mode_vacation():
    if set_system_mode('vacation'):
        return jsonify({'success': True, 'mode': 'vacation'})
    return jsonify({'success': False, 'message': 'Failed to set mode'}), 500

@app.route('/api/mode/request-otp', methods=['POST'])
def request_mode_otp():
    try:
        email = RECEIVER_EMAIL.lower()
        otp = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.now() + timedelta(minutes=10)
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO mode_otps (email, otp, expires_at, verified)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(email) DO UPDATE SET
                otp=excluded.otp,
                expires_at=excluded.expires_at,
                verified=0
        ''', (email, otp, expires_at.isoformat()))
        conn.commit()
        conn.close()

        subject = "SmartLock Mode Change - OTP Verification"
        body = f"""Dear User,

You have requested to switch SmartLock to In-Town mode.

Your One-Time Password (OTP) is: {otp}

This OTP is valid for 10 minutes only.

If you did not request this, please ignore this email.

Security Notice: Never share this OTP with anyone.

Best regards,
SmartLock System"""
        try:
            send_email(subject, body, to_email=email)
            return jsonify({'success': True, 'message': f'OTP sent to {email}', 'email': email}), 200
        except Exception as email_err:
            print(f"[ERROR] Failed to send mode OTP email: {email_err}")
            return jsonify({'success': False, 'message': 'Failed to send OTP email'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mode/verify-otp', methods=['POST'])
def verify_mode_otp():
    try:
        email = RECEIVER_EMAIL.lower()
        data = request.get_json()
        otp_input = data.get('otp', '').strip()
        if not otp_input:
            return jsonify({'success': False, 'message': 'OTP is required'}), 400
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('SELECT otp, expires_at FROM mode_otps WHERE email=?', (email,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return jsonify({'success': False, 'message': 'No OTP request found. Please request OTP first.'}), 400
        stored_otp, expires_at_str = row
        expires_at = datetime.fromisoformat(expires_at_str)
        if datetime.now() > expires_at:
            cursor.execute('DELETE FROM mode_otps WHERE email=?', (email,))
            conn.commit()
            conn.close()
            return jsonify({'success': False, 'message': 'OTP has expired. Please request a new one.'}), 400
        if stored_otp != otp_input:
            conn.close()
            return jsonify({'success': False, 'message': 'Incorrect OTP. Please try again.'}), 400
        cursor.execute('UPDATE mode_otps SET verified=1 WHERE email=?', (email,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'OTP verified successfully', 'email': email}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mode/in-town', methods=['POST'])
def set_mode_in_town():
    try:
        email = RECEIVER_EMAIL.lower()
        conn = sqlite3.connect('smart_lock_events.db')
        cursor = conn.cursor()
        cursor.execute('SELECT verified FROM mode_otps WHERE email=?', (email,))
        row = cursor.fetchone()
        if not row or not row[0]:
            conn.close()
            return jsonify({'success': False, 'message': 'OTP verification required. Please verify OTP first.'}), 403
        if not set_system_mode('in_town'):
            conn.close()
            return jsonify({'success': False, 'message': 'Failed to set mode'}), 500
        cursor.execute('DELETE FROM mode_otps WHERE email=?', (email,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'mode': 'in_town'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/password/request-otp', methods=['POST'])
def request_password_otp():
    """Request OTP for password change - sends to email"""
    try:
        # Use pre-configured email from alert system
        email = RECEIVER_EMAIL.lower()
        
        # Generate 6-digit OTP
        otp = ''.join(random.choices(string.digits, k=6))
        expires_at = datetime.now() + timedelta(minutes=10)  # OTP valid for 10 minutes
        
        # Store OTP in memory
        password_otps[email] = {
            'otp': otp,
            'expires_at': expires_at,
            'verified': False
        }
        
        # Send OTP via email
        subject = "SmartLock Password Change - OTP Verification"
        body = f"""Dear User,

You have requested to change your SmartLock PIN. 

Your One-Time Password (OTP) is: {otp}

This OTP is valid for 10 minutes only.

If you did not request this, please ignore this email.

Security Notice: Never share this OTP with anyone.

Best regards,
SmartLock System"""
        
        try:
            send_email(subject, body, to_email=email)
            print(f"[OK] OTP sent to {email}")
            
            return jsonify({
                'success': True,
                'message': f'OTP sent to {email}',
                'email': email
            }), 200
        except Exception as email_err:
            print(f"[ERROR] Failed to send OTP email: {email_err}")
            return jsonify({
                'success': False,
                'message': 'Failed to send OTP email'
            }), 500
            
    except Exception as e:
        print(f"[ERROR] OTP request error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/password/verify-otp', methods=['POST'])
def verify_password_otp():
    """Verify OTP for password change"""
    try:
        email = RECEIVER_EMAIL.lower()
        data = request.get_json()
        otp_input = data.get('otp', '').strip()
        
        if not otp_input:
            return jsonify({
                'success': False,
                'message': 'OTP is required'
            }), 400
        
        # Check if OTP exists and is valid
        if email not in password_otps:
            return jsonify({
                'success': False,
                'message': 'No OTP request found. Please request OTP first.'
            }), 400
        
        otp_data = password_otps[email]
        
        # Check if OTP has expired
        if datetime.now() > otp_data['expires_at']:
            del password_otps[email]
            return jsonify({
                'success': False,
                'message': 'OTP has expired. Please request a new one.'
            }), 400
        
        # Verify OTP
        if otp_data['otp'] != otp_input:
            return jsonify({
                'success': False,
                'message': 'Incorrect OTP. Please try again.'
            }), 400
        
        # Mark as verified
        otp_data['verified'] = True
        
        print(f"[OK] OTP verified for {email}")
        
        return jsonify({
            'success': True,
            'message': 'OTP verified successfully',
            'email': email
        }), 200
        
    except Exception as e:
        print(f"[ERROR] OTP verification error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/password/change', methods=['POST'])
def change_password():
    """Change PIN after OTP verification"""
    try:
        email = RECEIVER_EMAIL.lower()
        data = request.get_json()
        new_pin = data.get('new_pin', '').strip()
        confirm_pin = data.get('confirm_pin', '').strip()
        
        if not new_pin or not confirm_pin:
            return jsonify({
                'success': False,
                'message': 'New PIN and confirmation are required'
            }), 400
        
        # Check if OTP was verified
        if email not in password_otps or not password_otps[email]['verified']:
            return jsonify({
                'success': False,
                'message': 'OTP verification required. Please verify OTP first.'
            }), 403
        
        # Validate PIN format
        if len(new_pin) < 4:
            return jsonify({
                'success': False,
                'message': 'PIN must be at least 4 digits'
            }), 400
        
        if len(new_pin) > 8:
            return jsonify({
                'success': False,
                'message': 'PIN must be at most 8 digits'
            }), 400
        
        if not new_pin.isdigit():
            return jsonify({
                'success': False,
                'message': 'PIN must contain only digits'
            }), 400
        
        # Verify PINs match
        if new_pin != confirm_pin:
            return jsonify({
                'success': False,
                'message': 'PINs do not match. Please try again.'
            }), 400
        
        # Update PIN using secure hash storage
        try:
            result = PINSecurityManager.save_pin_hash(new_pin)
            
            if not result['success']:
                return jsonify({
                    'success': False,
                    'message': f"PIN change failed: {result.get('error', 'Unknown error')}"
                }), 400
            
            # Update global PIN hash
            global SYSTEM_PIN_DATA, SYSTEM_PIN_HASH
            SYSTEM_PIN_DATA = PINSecurityManager.load_pin_hash()
            SYSTEM_PIN_HASH = SYSTEM_PIN_DATA.get('pin_hash', '') if SYSTEM_PIN_DATA else ''
            
            # Clean up OTP record
            del password_otps[email]
            
            print(f"[OK] PIN changed successfully for {email}")
            
            # Send confirmation email
            subject = "SmartLock PIN Changed Successfully"
            confirmation_body = """Dear User,

Your SmartLock PIN has been changed successfully.

If you did not make this change, please contact the administrator immediately.

Best regards,
SmartLock System"""
            
            try:
                send_email(subject, confirmation_body, to_email=email)
            except Exception as conf_err:
                print(f"[WARNING] Failed to send confirmation email: {conf_err}")
            
            return jsonify({
                'success': True,
                'message': 'PIN changed successfully. Please use your new PIN to access the system.'
            }), 200
            
        except Exception as file_err:
            print(f"[ERROR] Failed to save new PIN: {file_err}")
            return jsonify({
                'success': False,
                'message': 'Failed to update PIN. Please try again.'
            }), 500
        
    except Exception as e:
        print(f"[ERROR] PIN change error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        stats = db_logger.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_behavior_score(datetime_str, action, face_score=0.0, voice_score=0.0, pin_valid=False):
    """Calculate behavior score based on time and biometric quality."""
    if not datetime_str:
        return 0.5  # Neutral score if no time provided
    
    try:
        dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        hour = dt.hour
        weekday = dt.weekday()  # 0=Mon ... 5=Sat ... 6=Sun
        
        # Base score by time window
        # Normal behavior: 7 AM - 10 PM
        if 7 <= hour <= 22:
            base_score = 0.9
        # Late night/early morning (less common)
        elif 23 <= hour or hour <= 6:
            # Late night: do not penalize on Saturday
            base_score = 0.9 if weekday == 5 else 0.6
        else:
            base_score = 0.7

        # Penalize weak biometrics and invalid PIN
        quality_penalty = 0.0
        if face_score < 0.75:
            quality_penalty += 0.10
        if voice_score < 0.75:
            quality_penalty += 0.10
        if not pin_valid:
            quality_penalty += 0.10

        # Extra penalty when weak biometrics happen at unusual hours
        if (hour >= 23 or hour <= 6) and weekday != 5 and (face_score < 0.75 or voice_score < 0.75):
            quality_penalty += 0.10

        return max(0.3, min(0.95, base_score - quality_penalty))
    except:
        return 0.5
    
@app.route('/api/debug_upload', methods=['POST'])
def debug_upload():
    """Debug file uploads"""
    print("\n" + "="*50)
    print("[SEARCH] DEBUG UPLOAD - What we're receiving:")
    print("="*50)
    
    print("Form data:", dict(request.form))
    
    files_received = {}
    for key, file in request.files.items():
        print(f"\n[FOLDER] File '{key}':")
        print(f"  Filename: {file.filename}")
        print(f"  Content-Type: {file.content_type}")
        print(f"  Size: {len(file.read())} bytes")
        file.seek(0)  # Reset pointer
        
        # Save file temporarily for inspection
        temp_path = os.path.join(tempfile.gettempdir(), f"debug_{key}_{datetime.now().timestamp()}")
        file.save(temp_path)
        print(f"  Saved to: {temp_path}")
        
        files_received[key] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(file.read()),
            'temp_path': temp_path
        }
        file.seek(0)
    
    print("="*50)
    
    return jsonify({
        'form_data': dict(request.form),
        'files': files_received,
        'message': 'Debug info printed to console'
    })

@app.route('/api/test_connection')
def test_connection():
    """Test if backend is reachable"""
    return jsonify({
        'status': 'online',
        'message': 'Backend is reachable',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/enroll_face', methods=['POST'])
def enroll_face():
    """Enroll a new face (temporary for testing)"""
    try:
        if 'face_image' not in request.files:
            return jsonify({'error': 'No face image provided'}), 400
        
        face_file = request.files['face_image']
        user_id = request.form.get('user_id', 'owner')
        
        if face_module is None:
            return jsonify({'error': 'Face module not initialized'}), 500
        
        # Save temp file
        tmp_dir = tempfile.mkdtemp()
        face_path = os.path.join(tmp_dir, 'enroll_face.jpg')
        face_file.save(face_path)
        
        # Get face embedding
        embedding = face_module['get_face_embedding'](face_path)
        if embedding is not None:
            # Add to database
            face_module['face_db'][user_id] = embedding
            
            # Save database
            with open(face_module['db_path'], 'wb') as f:
                pickle.dump(face_module['face_db'], f)
            
            # Cleanup
            os.remove(face_path)
            os.rmdir(tmp_dir)
            
            return jsonify({
                'success': True,
                'message': f'Face enrolled for {user_id}',
                'face_count': len(face_module['face_db'])
            })
        else:
            return jsonify({'error': 'Failed to extract face embedding'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test_biometrics')
def test_biometrics():
    """Test if biometric modules are working"""
    test_results = {}
    
    # Test face module
    if face_module:
        test_results['face_module'] = {
            'status': 'loaded',
            'faces_in_db': len(face_module['face_db']),
            'functions': list(face_module.keys())
        }
    else:
        test_results['face_module'] = {'status': 'not_loaded'}
    
    # Test voice module
    if voice_module:
        test_results['voice_module'] = {
            'status': 'loaded',
            'has_verifier': 'verifier' in voice_module,
            'simulated': voice_module.get('simulated', False)
        }
    else:
        test_results['voice_module'] = {'status': 'not_loaded'}
    
    return jsonify(test_results)

@app.route("/api/unlock", methods=["POST"])
def unlock():
    """Old-style endpoint with background processing"""
    if 'voice_audio' not in request.files or 'face_image' not in request.files:
        return jsonify({"error": "Missing files"}), 400
        
    pin = request.form.get("pin", "").strip()
    audio_file = request.files["voice_audio"]
    image_file = request.files["face_image"]

    tmp_dir = tempfile.mkdtemp()
    a_path = os.path.join(tmp_dir, "v.webm")
    i_path = os.path.join(tmp_dir, "f.jpg")
    
    audio_file.save(a_path)
    image_file.save(i_path)

    task_id = str(uuid.uuid4())
    q = Queue()
    progress_queues[task_id] = (q, tmp_dir)
    
    # Start background thread (using your old function)
    threading.Thread(
        target=process_attempt_background, 
        args=(pin, a_path, i_path, q, tmp_dir)
    ).start()
    
    return jsonify({"task_id": task_id})

@app.route("/api/progress/<task_id>")
def api_progress(task_id):
    """SSE stream for progress updates"""
    def stream():
        item = progress_queues.get(task_id)
        if not item: 
            yield f"data: {json.dumps({'error': 'Invalid task'})}\n\n"
            return
            
        q, tmp_dir = item
        while True:
            msg = q.get()
            if msg is None: 
                break
            yield f"data: {json.dumps(msg)}\n\n"
        
        # Cleanup
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        progress_queues.pop(task_id, None)
        
    return Response(stream(), mimetype="text/event-stream")

# Old route for compatibility
@app.route("/progress/<task_id>")
def progress(task_id):
    def stream():
        item = progress_queues.get(task_id)
        if not item: return
        q, tmp_dir = item
        while True:
            msg = q.get()
            if msg is None: break
            yield f"data: {json.dumps(msg)}\n\n"
        
        # Clean up temp folder after the stream closes
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        progress_queues.pop(task_id, None)
        
    return Response(stream(), mimetype="text/event-stream")



if __name__ == '__main__':
    print("\n" + "="*60)
    print("[START] INITIALIZING GENAI SMART LOCK SYSTEM")
    print("="*60)
    
#   Run diagnostics FIRST
    diagnose_imports()  # <-- ADD THIS LINE
    
    # Initialize components
    print("\n[CONFIG] Initializing components...")

    # Initialize components
    print("\n[CONFIG] Initializing components...")
    db_logger = EventLogger()
    print("[OK] Database initialized")
    
    ai_analyzer = GenAIAnalyzer()
    print("[OK] GenAI analyzer initialized")
    
    face_module = init_face_module()
    voice_module = init_voice_module()
    
    # Check module initialization
    if face_module:
        print(f"[OK] FaceNet loaded successfully with {len(face_module['face_db'])} faces")
    else:
        print("[WARNING]  FaceNet module initialization failed")
    
    if voice_module:
        status = "Simulated" if voice_module.get('simulated', False) else "Real"
        print(f"[OK] VoiceRecognition loaded successfully ({status})")
    else:
        print("[WARNING]  VoiceRecognition module initialization failed")
    
    print("\n" + "="*60)
    print("[OK] SYSTEM READY")
    print(f"[SECURITY] PIN: {'***' if SYSTEM_PIN_HASH else 'Not configured'} [Secured with SHA-256 Hashing]")
    print(f"[WEB] Web Interface: http://localhost:5000")
    print(f"[AI] GenAI: {'Connected' if ai_analyzer.api_key else 'Simulation Mode'}")
    print("="*60 + "\n")
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('embeddings', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
