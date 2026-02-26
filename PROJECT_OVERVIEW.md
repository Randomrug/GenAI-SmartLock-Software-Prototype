# GenAI Smart Lock System - Complete Project Overview

## Project Description
This is a **research-grade smart lock access control system** that uses:
- **Biometric Authentication** (Face Recognition + Voice Recognition)
- **PIN-based Authentication**
- **Generative AI** (OpenRouter API with LLaMA 3.3-70B or other free models)
- **Machine Learning** (Anomaly Detection, Behavior Modeling, Score Fusion)
- **SQLite Database** for persistent event logging
- **Web Interface** (Flask + Modern HTML/CSS/JS)

## Project Structure

### Root Level Files
| File | Purpose |
|------|---------|
| `run.py` | Main entry point launcher with comprehensive system initialization |
| `app.py` | Flask web application (562 lines) - Routes, authentication endpoints |
| `ai_safety.py` | GenAI analyzer using OpenRouter API (572 lines) |
| `event_logger.py` | SQLite database interface for logging access events (552 lines) |
| `pin.txt` | Stores system PIN (default: 1234) |
| `requirements.txt` | Python dependencies (67 lines) |

---

## Module Breakdown

### 1. **Configuration/** - System Configuration Management
**Files:**
- `config.py` (396 lines) - Central configuration manager with defaults for server, database, authentication, security, AI, logging, paths, behavior, performance
- `security_config.py` (517 lines) - Security-specific settings: lockout rules, PIN validation, biometric thresholds, session management, rate limiting, encryption, audit logging
- `ai_config.py` (545 lines) - AI provider configuration (OpenRouter, OpenAI, Anthropic, Local), model parameters, prompt templates, ML model configurations

**Key Features:**
- Centralized config management with JSON file support
- Deep configuration merging
- Encryption key management
- Multi-provider AI support

---

### 2. **FaceNet/** - Face Recognition Module
**Files:**
- `face_embedder.py` (40 lines) - Uses FaceNet-PyTorch to generate face embeddings
  - Model: InceptionResnetV1 (VGGFace2 pretrained)
  - Face Detection: MTCNN
  - Output: Normalized embedding vectors
  
- `verify_face.py` (15 lines) - Face verification using cosine similarity
  - Compares live face embedding against face database
  - Returns owner name, similarity score, and verification result
  
- `enroll_faces.py` (40 lines) - Face enrollment system
  - Reads face images from `/enroll/{owner_name}/` directory
  - Generates embeddings and stores as pickle file
  - Supports multiple images per person (averaged)
  
- `capture_live.py` (40 lines) - Real-time face capture
  - Uses OpenCV to access webcam
  - 3-second countdown before capture
  - ESC key to cancel

**Configuration:**
- Stored in `/embeddings/face_db.pkl`
- Sample faces in `/enroll/rithika/`
- Threshold: 0.7 (70% similarity required)

---

### 3. **VoiceRecognition/** - Voice Authentication Module
**Files:**
- `smart_lock.py` (159 lines) - Voice verification endpoint
  - Uses SpeechBrain speaker recognition (ECAPA-VoxCeleb)
  - 5-second voice recording required
  - Max 3 attempts before lockout
  - Integrates with FaceNet for two-factor verification
  - Phrase: "Hey lock, open the door please"
  
- `enroll_my_voice.py` (55 lines) - Voice enrollment
  - Records 5 samples of user's voice
  - Creates averaged voice template stored as PyTorch tensor
  - Template saved to `/templates/my_voice_template.pt`

**Pretrained Models:**
- Location: `/pretrained_models/spkrec-ecapa-voxceleb/`
- Contains: classifier.ckpt, embedding_model.ckpt, hyperparams.yaml, label_encoder.ckpt, mean_var_norm_emb.ckpt
- Threshold: 0.65 (65% cosine similarity)

---

### 4. **Models/** - Machine Learning Module
**Files:**

#### `anomaly_detector.py` (553 lines)
- **Purpose:** Detects unusual access patterns using ML algorithms
- **Algorithms:**
  - Isolation Forest (100 estimators, 10% contamination)
  - One-Class SVM (RBF kernel)
  - Local Outlier Factor (20 neighbors)
- **Features Extracted:**
  - Temporal: hour, day_of_week, is_weekend, circular encoding
  - Behavioral: face_score, voice_score, final_score, pin_valid
  - Contextual: failed_attempts, risk_level, time_since_last_access
- **Methods:** detect(), train_models(), load_models(), save_models()

#### `behavior_model.py` (832 lines)
- **Purpose:** Models and predicts normal user behavior
- **Techniques:**
  - K-Means clustering (3 clusters default)
  - DBSCAN for density-based analysis
  - Gaussian Mixture Models
  - PCA for dimensionality reduction
- **Pattern Detection:**
  - Daily patterns (hour-based)
  - Weekly patterns (day-of-week)
  - Access type patterns (LOCK vs UNLOCK)
  - Time since last access
- **Output:** Behavior scores and anomaly flags

#### `score_fusion.py` (727 lines)
- **Purpose:** Intelligently combines multiple authentication scores
- **Methods:**
  1. Weighted Average (default weights: PIN 0.3, Face 0.35, Voice 0.35)
  2. Machine Learning fusion:
     - Random Forest (100 estimators)
     - Gradient Boosting (100 estimators)
     - SVM with RBF kernel
  3. Rule-based fusion:
     - Minimum threshold checking
     - Consistency validation
     - Context adjustment
- **Calibration:** Isotonic or Sigmoid calibration
- **Final Threshold:** 0.7 (70% required for ALLOW)

---

### 5. **Utilities/** - Administrative Tools & Testing
**Files:**

#### `admin_tools.py` (376 lines)
- Reset lockout states
- View system logs with filtering (lockout, denied, allowed, recent)
- Export logs to JSON or CSV format
- Statistics calculation
- Admin authentication (hardcoded key: "admin123")

#### `test_suite.py` (565 lines)
- Unit tests for database functionality
- GenAI analyzer tests
- Test database creation
- Event logging tests
- Failure streak tracking tests
- Statistics calculation tests

#### `database_seeder.py`
- Pre-seeds database with synthetic historical data (100 synthetic events)
- Generates realistic access patterns

#### `backup_database.py`
- Database backup functionality

---

### 6. **Static Assets/**
**CSS:**
- `dashboard.css` - Dashboard styling
- `style.css` - General styling

**JavaScript:**
- `capture_face.js` - Face capture form handling
- `dashboard.js` - Dashboard interactions and charts
- `record_voice.js` - Voice recording functionality

---

### 7. **Templates/**
- `index.html` (1326 lines) - Main web interface
  - Modern glassmorphism design
  - Responsive grid layout
  - Real-time system status
  - Access event logs with filtering
  - Authentication control panel
  - Statistics dashboard with charts
  - Admin panel with lockout reset

---

## Database Schema (SQLite)

### `access_events` Table
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Event ID |
| action | TEXT | CHECK('LOCK','UNLOCK') | Lock/Unlock action |
| entry_time | TEXT | | Unlock timestamp |
| exit_time | TEXT | | Lock timestamp |
| pin_valid | BOOLEAN | NOT NULL | PIN validation result |
| face_score | REAL | 0-1 range | Face similarity score |
| voice_score | REAL | 0-1 range | Voice similarity score |
| behavior_score | REAL | 0-1 range | Behavioral normalcy score |
| final_score | REAL | 0-1 range | Weighted final score |
| genai_decision | TEXT | ALLOW/DENY/LOCKOUT | AI decision |
| genai_risk_level | TEXT | LOW/MEDIUM/HIGH | Risk assessment |
| genai_explanation | TEXT | NOT NULL | Decision explanation |
| failed_attempt_count | INTEGER | DEFAULT 0 | Failure streak counter |
| lockout_active | BOOLEAN | DEFAULT 0 | Lockout status |
| record_created_at | TIMESTAMP | DEFAULT NOW | Creation timestamp |

**Indexes:**
- `idx_decision` - On (genai_decision, record_created_at)
- `idx_lockout` - On (lockout_active, record_created_at)
- `idx_timestamp` - On record_created_at DESC
- `idx_scores` - On (face_score, voice_score, final_score)

### `system_statistics` Table
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Stat ID |
| stat_name | TEXT UNIQUE | Statistic name |
| stat_value | TEXT | Statistic value |
| updated_at | TIMESTAMP | Last update |

---

## Authentication Flow

### Multi-Factor Authentication Chain:
```
1. PIN Entry (weight: 0.3)
   └─ Valid if matches stored PIN in pin.txt

2. Face Recognition (weight: 0.35)
   ├─ Capture live face via webcam
   ├─ Extract embedding using FaceNet
   ├─ Compare against face_db.pkl
   └─ Score: cosine similarity (threshold: 0.7)

3. Voice Recognition (weight: 0.35)
   ├─ Record 5-second voice sample
   ├─ Extract embedding using SpeechBrain
   ├─ Compare against my_voice_template.pt
   └─ Score: cosine similarity (threshold: 0.65)

4. Behavioral Analysis
   ├─ Check access time patterns
   ├─ Verify against historical behavior
   └─ Anomaly detection (Isolation Forest, LOF, One-Class SVM)

5. Final Score Fusion (weighted average)
   └─ PIN_score × 0.3 + Face_score × 0.35 + Voice_score × 0.35

6. GenAI Safety Analysis (via OpenRouter API)
   ├─ Build comprehensive context
   ├─ Include historical patterns
   ├─ Identify risk factors
   ├─ Consult LLaMA 3.3-70B
   └─ Decision: ALLOW / DENY / LOCKOUT

7. Lockout Management
   ├─ Track failed attempts
   ├─ Trigger lockout after 5 failures
   ├─ 30-minute lockout duration
   └─ Admin can reset with key
```

---

## Scoring System

### Component Scores
- **PIN Score:** 1.0 if valid, 0.0 if invalid
- **Face Score:** 0.0 - 1.0 (cosine similarity)
- **Voice Score:** 0.0 - 1.0 (cosine similarity)
- **Behavior Score:** 0.6 - 0.9 based on time-of-day patterns

### Final Score Calculation
```
final_score = (PIN_score × 0.3) + (Face_score × 0.35) + (Voice_score × 0.35)
```

### Risk Level Assignment
- **LOW:** final_score ≥ 0.8, no anomalies detected
- **MEDIUM:** 0.6 ≤ final_score < 0.8, minor anomalies
- **HIGH:** final_score < 0.6 or major anomalies detected

---

## GenAI Integration (OpenRouter API)

### Configuration
- **Provider:** OpenRouter (free tier)
- **Default Model:** meta-llama/llama-3.3-70b-instruct:free
- **Alternative Models:**
  - google/gemma-2-9b-it:free
  - mistralai/mistral-7b-instruct:free
  - qwen/qwen-2.5-32b-instruct:free
- **API Timeout:** 30 seconds
- **Max Retries:** 3

### AI Analysis Process
1. Build comprehensive context including:
   - Current attempt data
   - Historical patterns
   - Recent events (last 15)
   - Statistics (success rate, avg scores, failure streak)
   - Risk factors identification
   - System state

2. Construct detailed prompt with:
   - Current biometric scores
   - PIN validity
   - Historical access patterns
   - Time-based anomalies
   - Device/location history

3. GenAI response parsing:
   - Extracts decision (ALLOW/DENY/LOCKOUT)
   - Risk level assessment (LOW/MEDIUM/HIGH)
   - Detailed explanation of reasoning

### Fallback Mode
- If API fails or no API key: Uses rule-based analysis
- Simple logic: final_score ≥ 0.7 = ALLOW, else DENY

---

## Web Interface Features

### Main Sections
1. **System Status Panel**
   - Real-time system status (online/offline)
   - Total access events
   - Success/Deny/Lockout counts
   - Module availability (Face, Voice, AI)

2. **Authentication Control Panel**
   - Manual date/time input
   - Lock/Unlock action selection
   - PIN entry
   - Face image capture
   - Voice recording
   - Submit authentication request

3. **Real-time Results Display**
   - Decision (ALLOW/DENY/LOCKOUT)
   - Risk level
   - AI explanation
   - Component scores (PIN, Face, Voice, Behavior, Final)
   - Failed attempts counter

4. **Access Event Log**
   - Real-time event feed (latest first)
   - Filtering by decision type
   - Time sorting
   - Color-coded decisions (✅ ALLOW, ❌ DENY, 🔒 LOCKOUT)
   - Detailed event information

5. **Statistics Dashboard**
   - Success/Failure rate charts
   - Average scores visualization
   - Risk level distribution
   - Time-based access patterns

6. **Admin Panel**
   - Lockout status monitoring
   - Manual lockout reset
   - Admin key authentication

---

## Security Features

### Implemented
- PIN-based authentication (4-8 digits)
- Biometric multi-factor authentication
- Failed attempt tracking (max 5 failures)
- Automatic lockout (30 minutes)
- Encryption key storage in `/keys/`
- Audit logging (all events recorded)
- GenAI-powered behavioral analysis
- Anomaly detection (multiple algorithms)

### Configuration Options
- Configurable security thresholds
- Customizable lockout duration
- Admin-reset capability
- IP whitelisting (disabled by default)
- Session management (30-minute timeout)
- HTTPS enforcement option

---

## Deployment & Execution

### Installation
```bash
pip install -r requirements.txt
```

### Running the System
```bash
# Option 1: Using run.py (launcher)
python run.py

# Option 2: Direct Flask app
python app.py

# Access web interface
http://localhost:5000
```

### Voice Enrollment (One-time)
```bash
python VoiceRecognition/enroll_my_voice.py
```

### Face Enrollment
1. Place face images in `/enroll/rithika/` directory
2. Run: `python FaceNet/enroll_faces.py`

### Testing
```bash
python Utilities/test_suite.py
```

---

## API Endpoints

### Authentication
- `POST /api/authenticate` - Main authentication endpoint
  - Parameters: action, pin, face_image, voice_audio, manual_date, manual_time
  - Returns: decision, risk_level, scores, explanation

### System Information
- `GET /api/status` - System status
- `GET /api/events?limit=50` - Recent access events
- `GET /api/stats` - System statistics

### Administration
- `POST /api/reset_lockout` - Reset lockout state
  - Parameters: admin_key

---

## File Organization Summary

```
LOCK/
├── Root Files
│   ├── run.py                 (Launcher)
│   ├── app.py                 (Flask app)
│   ├── ai_safety.py           (GenAI analyzer)
│   ├── event_logger.py        (Database)
│   ├── pin.txt                (System PIN)
│   └── requirements.txt       (Dependencies)
│
├── Configuration/
│   ├── config.py              (Main config)
│   ├── security_config.py     (Security settings)
│   └── ai_config.py           (AI config)
│
├── FaceNet/
│   ├── face_embedder.py       (Embedding generator)
│   ├── verify_face.py         (Face verification)
│   ├── enroll_faces.py        (Enrollment)
│   ├── capture_live.py        (Live capture)
│   ├── embeddings/            (Stored embeddings)
│   └── enroll/                (Enrollment images)
│
├── VoiceRecognition/
│   ├── smart_lock.py          (Voice verification)
│   ├── enroll_my_voice.py     (Enrollment)
│   ├── pretrained_models/     (SpeechBrain models)
│   └── templates/             (Voice templates)
│
├── Models/
│   ├── anomaly_detector.py    (Anomaly detection)
│   ├── behavior_model.py      (Behavior modeling)
│   └── score_fusion.py        (Score fusion)
│
├── Utilities/
│   ├── admin_tools.py         (Admin functions)
│   ├── test_suite.py          (Tests)
│   ├── database_seeder.py     (Data seeding)
│   └── backup_database.py     (Backups)
│
├── Static/
│   ├── css/                   (Stylesheets)
│   ├── js/                    (JavaScript)
│   └── assets/                (Images/files)
│
├── Templates/
│   └── index.html             (Web interface)
│
├── logs/                      (Log files)
├── keys/                      (Encryption keys)
└── smart_lock_events.db       (SQLite database)
```

---

## Key Technologies & Libraries

### Core
- Python 3.x
- Flask (Web framework)
- SQLite3 (Database)

### Biometric
- facenet-pytorch (Face recognition)
- MTCNN (Face detection)
- SpeechBrain (Voice recognition)
- librosa (Audio processing)
- OpenCV (Camera access)

### AI/ML
- scikit-learn (Anomaly detection, ML models)
- NumPy, Pandas (Data processing)
- torch, torchvision (Deep learning)
- requests (API calls)

### Utilities
- werkzeug (File handling)
- joblib (Model serialization)
- cryptography (Encryption)

---

## Performance Considerations

### Thresholds
- Face similarity: 0.7 (70%)
- Voice similarity: 0.65 (65%)
- Final combined score: 0.7 (70%)

### Timeouts
- GenAI API: 30 seconds
- Session: 30 minutes

### Database
- 4 performance indexes
- Max 1000 query results
- Connection pool: 5 connections

### Anomaly Detection
- Isolation Forest: 100 trees, 10% contamination
- LOF: 20 neighbors
- Training samples: 100+

---

## Security Considerations

- **API Key Management:** OpenRouter API key stored in environment variables
- **Admin Access:** Hardcoded admin key (should be changed in production)
- **Encryption:** AES-256-GCM for sensitive data
- **Audit Trail:** All access attempts logged to database
- **Rate Limiting:** Optional IP-based rate limiting
- **Session Management:** 30-minute timeout with renewal intervals

---

## Future Enhancement Opportunities

1. User management system (multiple users)
2. Email/SMS alerts on security events
3. Geolocation-based authentication
4. Liveness detection for face authentication
5. Advanced time-series forecasting
6. Real-time dashboard with WebSocket updates
7. Mobile app integration
8. Biometric template protection (fuzzy vault)
9. Hardware integration (actual smart lock)
10. Advanced encryption for at-rest data

---

**Project Status:** Research/Development Grade
**Last Updated:** January 15, 2026
**Author:** Rithika
