/**
 * Webcam Face Capture for Smart Lock System
 * Captures face image from webcam and converts to base64
 */

class FaceCapture {
    constructor() {
        this.video = document.getElementById('faceVideo');
        this.canvas = document.getElementById('faceCanvas');
        this.preview = document.getElementById('facePreview');
        this.status = document.getElementById('faceStatus');
        this.captureBtn = document.getElementById('captureFace');
        
        this.stream = null;
        this.capturedImage = null;
        
        this.init();
    }
    
    init() {
        // Check for webcam support
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Webcam not supported by your browser');
            return;
        }
        
        this.captureBtn.addEventListener('click', () => this.toggleCamera());
        
        // Request camera permissions on page load (optional)
        // this.startCamera();
    }
    
    async toggleCamera() {
        if (this.stream) {
            this.capturePhoto();
        } else {
            await this.startCamera();
        }
    }
    
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user' // Front camera
                } 
            });
            
            this.video.srcObject = this.stream;
            this.video.style.display = 'block';
            
            // Wait for video to be ready
            await new Promise(resolve => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });
            
            this.status.innerHTML = '📷 Camera active - Look at the camera';
            this.status.style.color = '#4CAF50';
            this.captureBtn.innerHTML = '📸 Capture Face';
            this.captureBtn.style.background = 'linear-gradient(135deg, #ff416c, #ff4b2b)';
            
            console.log('✅ Camera started successfully');
            
        } catch (error) {
            this.handleCameraError(error);
        }
    }
    
    capturePhoto() {
        if (!this.stream) return;
        
        // Set canvas dimensions to match video
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        
        // Draw current video frame to canvas
        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert to data URL (JPEG format, 80% quality)
        const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
        this.capturedImage = imageData;
        
        // Show preview
        this.showPreview(imageData);
        
        // Stop camera
        this.stopCamera();
        
        // Update button
        this.captureBtn.innerHTML = '📷 Capture Face';
        this.captureBtn.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
        
        // Notify parent
        if (typeof window.onFaceCaptured === 'function') {
            window.onFaceCaptured(imageData);
        }
        
        console.log('✅ Face captured successfully');
    }
        capturePhoto() {
            if (!this.stream) return;
            // Set canvas dimensions to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
            // Draw current video frame to canvas
            const ctx = this.canvas.getContext('2d');
            ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            // Convert to Blob (JPEG format, 80% quality)
            this.canvas.toBlob((blob) => {
                this.capturedBlob = blob;
                const imageData = this.canvas.toDataURL('image/jpeg', 0.8);
                this.capturedImage = imageData;
                this.showPreview(imageData);
                this.stopCamera();
                this.captureBtn.innerHTML = '📷 Capture Face';
                this.captureBtn.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
                if (typeof window.onFaceCaptured === 'function') {
                    window.onFaceCaptured(blob);
                }
                console.log('✅ Face captured successfully');
            }, 'image/jpeg', 0.8);
        }
    
    showPreview(imageData) {
        this.preview.innerHTML = `
            <div style="text-align: center;">
                <img src="${imageData}" 
                     style="max-width: 100%; max-height: 200px; border-radius: 10px; border: 3px solid #4CAF50;">
                <p style="margin-top: 10px; color: #4CAF50;">
                    ✅ Face captured successfully
                </p>
                <button onclick="faceCapture.retakePhoto()" 
                        style="margin-top: 5px; padding: 5px 15px; background: #ff416c; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    Retake
                </button>
            </div>
        `;
    }
    
    retakePhoto() {
        this.preview.innerHTML = '<p id="faceStatus">No face captured</p>';
        this.capturedImage = null;
        this.status = document.getElementById('faceStatus');
        this.startCamera();
    }
    
    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.video.style.display = 'none';
        }
    }
    
    handleCameraError(error) {
        console.error('Camera error:', error);
        
        let errorMessage = 'Cannot access camera: ';
        
        switch(error.name) {
            case 'NotFoundError':
            case 'DevicesNotFoundError':
                errorMessage += 'No camera found on this device.';
                break;
            case 'NotAllowedError':
            case 'PermissionDeniedError':
                errorMessage += 'Camera permission denied. Please allow camera access.';
                break;
            case 'NotReadableError':
            case 'TrackStartError':
                errorMessage += 'Camera is already in use by another application.';
                break;
            default:
                errorMessage += error.message || 'Unknown error.';
        }
        
        this.showError(errorMessage);
    }
    
    showError(message) {
        this.status.innerHTML = `❌ ${message}`;
        this.status.style.color = '#ff416c';
        this.captureBtn.disabled = true;
        this.captureBtn.style.opacity = '0.5';
    }
    
    getCapturedImage() {
        return this.capturedImage;
    }
        getCapturedImage() {
            return this.capturedBlob || null;
        }
    
    clearCapture() {
        this.capturedImage = null;
        this.preview.innerHTML = '<p id="faceStatus">No face captured</p>';
        this.status = document.getElementById('faceStatus');
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.faceCapture = new FaceCapture();
});