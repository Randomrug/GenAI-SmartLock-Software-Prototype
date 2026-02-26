/**
 * Voice Recording for Smart Lock System
 * Records audio from microphone and converts to base64
 */

class VoiceRecorder {
    constructor() {
        this.recordBtn = document.getElementById('recordVoice');
        this.preview = document.getElementById('voicePreview');
        this.status = document.getElementById('voiceStatus');
        this.playback = document.getElementById('voicePlayback');
        
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordedAudio = null;
        this.recordingTimer = null;
        this.recordingTime = 0;
        
        this.MAX_RECORDING_TIME = 10000; // 10 seconds max
        
        this.init();
    }
    
    init() {
        // Check for microphone support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Microphone not supported by your browser');
            return;
        }
        
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        
        // Initialize playback element
        if (this.playback) {
            this.playback.style.display = 'none';
        }
    }
    
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        try {
            // Request microphone permission
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000 // Match speech recognition requirements
                }
            });
            
            // Create MediaRecorder
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            // Reset chunks
            this.audioChunks = [];
            this.recordingTime = 0;
            
            // Setup data handler
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            // Setup stop handler
            this.mediaRecorder.onstop = () => {
                this.processRecording();
                
                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            };
            
            // Start recording
            this.mediaRecorder.start(100); // Collect data every 100ms
            
            this.isRecording = true;
            this.updateUIForRecording();
            
            // Start timer
            this.startRecordingTimer();
            
            // Auto-stop after max time
            setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording();
                }
            }, this.MAX_RECORDING_TIME);
            
            console.log('✅ Recording started');
            
        } catch (error) {
            this.handleRecordingError(error);
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.stopRecordingTimer();
            this.updateUIForStopped();
            console.log('✅ Recording stopped');
        }
    }
    
    startRecordingTimer() {
        this.recordingTime = 0;
        this.recordingTimer = setInterval(() => {
            this.recordingTime += 1;
            this.updateTimerDisplay();
        }, 1000);
    }
    
    stopRecordingTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }
    
    updateTimerDisplay() {
        if (this.status) {
            const seconds = this.recordingTime;
            this.status.innerHTML = `🎤 Recording... ${seconds}s (Max: 10s)`;
            this.status.style.color = '#ffcc00';
        }
    }
    
    async processRecording() {
        try {
            // Create audio blob
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            this.recordedBlob = audioBlob;
            // Convert to base64 (for preview/download only)
            const base64Audio = await this.blobToBase64(audioBlob);
            this.recordedAudio = base64Audio;
            // Show preview
            this.showPreview(audioBlob);
            // Notify parent
            if (typeof window.onVoiceRecorded === 'function') {
                window.onVoiceRecorded(audioBlob);
            }
            console.log('✅ Audio processed successfully');
        } catch (error) {
            console.error('Error processing recording:', error);
            this.showError('Failed to process audio recording');
        }
    }
    
    showPreview(audioBlob) {
        // Create object URL for playback
        const audioUrl = URL.createObjectURL(audioBlob);
        
        this.preview.innerHTML = `
            <div style="text-align: center;">
                <audio controls src="${audioUrl}" 
                       style="width: 100%; margin: 10px 0;">
                    Your browser does not support audio playback.
                </audio>
                <p style="color: #4CAF50; margin: 5px 0;">
                    ✅ Voice recorded successfully (${this.recordingTime}s)
                </p>
                <div style="margin-top: 10px;">
                    <button onclick="voiceRecorder.retakeRecording()" 
                            style="margin-right: 10px; padding: 5px 15px; background: #ff416c; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Retake
                    </button>
                    <button onclick="voiceRecorder.downloadRecording()" 
                            style="padding: 5px 15px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Download
                    </button>
                </div>
            </div>
        `;
    }
    
    retakeRecording() {
        this.recordedAudio = null;
        this.preview.innerHTML = '<p id="voiceStatus">No voice recorded</p>';
        this.status = document.getElementById('voiceStatus');
        this.recordBtn.innerHTML = '🎤 Record Voice';
        this.recordBtn.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
        
        console.log('🔄 Recording cleared');
    }
    
    downloadRecording() {
        if (!this.recordedAudio) return;
        
        // Convert base64 to blob
        const byteCharacters = atob(this.recordedAudio.split(',')[1]);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'audio/webm' });
        
        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `voice_recording_${new Date().toISOString().slice(0,19)}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
    
    updateUIForRecording() {
        this.recordBtn.innerHTML = '⏹ Stop Recording';
        this.recordBtn.style.background = 'linear-gradient(135deg, #ff416c, #ff4b2b)';
        
        if (this.status) {
            this.status.innerHTML = '🎤 Recording... Speak clearly into the microphone';
            this.status.style.color = '#ffcc00';
        }
    }
    
    updateUIForStopped() {
        this.recordBtn.innerHTML = '🎤 Record Voice';
        this.recordBtn.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
    }
    
    async blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }
    
    handleRecordingError(error) {
        console.error('Recording error:', error);
        
        let errorMessage = 'Cannot access microphone: ';
        
        switch(error.name) {
            case 'NotFoundError':
            case 'DevicesNotFoundError':
                errorMessage += 'No microphone found on this device.';
                break;
            case 'NotAllowedError':
            case 'PermissionDeniedError':
                errorMessage += 'Microphone permission denied. Please allow microphone access.';
                break;
            case 'NotReadableError':
            case 'TrackStartError':
                errorMessage += 'Microphone is already in use by another application.';
                break;
            default:
                errorMessage += error.message || 'Unknown error.';
        }
        
        this.showError(errorMessage);
    }
    
    showError(message) {
        if (this.status) {
            this.status.innerHTML = `❌ ${message}`;
            this.status.style.color = '#ff416c';
        }
        this.recordBtn.disabled = true;
        this.recordBtn.style.opacity = '0.5';
    }
    
    getRecordedAudio() {
        return this.recordedBlob || null;
    }
    
    clearRecording() {
        this.recordedAudio = null;
        this.preview.innerHTML = '<p id="voiceStatus">No voice recorded</p>';
        this.status = document.getElementById('voiceStatus');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.voiceRecorder = new VoiceRecorder();
});