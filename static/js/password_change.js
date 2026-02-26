    /**
 * Password Change Page - OTP Flow
 * Standalone page for secure PIN change
 */

class PasswordChangeForm {
    constructor() {
        this.currentStep = 1;
        this.otpVerified = false;
        this.email = 'rithikaarulmozhi21@gmail.com'; // Pre-configured from backend

        this.initializeElements();
        this.bindEvents();
        this.showStep(1);
    }

    initializeElements() {
        // Step containers
        this.step1 = document.getElementById('step1');
        this.step2 = document.getElementById('step2');
        this.step3 = document.getElementById('step3');
        this.step4 = document.getElementById('step4');

        // Buttons
        this.btnRequestOtp = document.getElementById('btnRequestOtp');
        this.btnVerifyOtp = document.getElementById('btnVerifyOtp');
        this.btnChangePin = document.getElementById('btnChangePin');
        this.btnDone = document.getElementById('btnDone');
        this.btnBackToStep1 = document.getElementById('btnBackToStep1');
        this.btnBackToStep2 = document.getElementById('btnBackToStep2');

        // Inputs
        this.otpInput = document.getElementById('otpInput');
        this.newPinInput = document.getElementById('newPinInput');
        this.confirmPinInput = document.getElementById('confirmPinInput');

        // Progress indicators
        this.progress1 = document.getElementById('progress1');
        this.progress2 = document.getElementById('progress2');
        this.progress3 = document.getElementById('progress3');

        // Message area
        this.messageArea = document.getElementById('messageArea');
    }

    bindEvents() {
        this.btnRequestOtp.addEventListener('click', () => this.requestOTP());
        this.btnVerifyOtp.addEventListener('click', () => this.verifyOTP());
        this.btnChangePin.addEventListener('click', () => this.changePin());
        this.btnDone.addEventListener('click', () => this.closeWindow());
        this.btnBackToStep1.addEventListener('click', () => this.showStep(1));
        this.btnBackToStep2.addEventListener('click', () => this.showStep(2));

        // Allow Enter key to submit
        this.otpInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.verifyOTP();
        });
        this.confirmPinInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.changePin();
        });
    }

    showMessage(message, type = 'info') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type} show`;
        
        let icon = '';
        switch(type) {
            case 'success': icon = '<i class="fas fa-check-circle"></i> '; break;
            case 'error': icon = '<i class="fas fa-times-circle"></i> '; break;
            case 'info': icon = '<i class="fas fa-info-circle"></i> '; break;
        }
        
        messageDiv.innerHTML = icon + message;
        this.messageArea.innerHTML = '';
        this.messageArea.appendChild(messageDiv);

        // Auto-remove after 5 seconds (except errors)
        if (type !== 'error') {
            setTimeout(() => {
                messageDiv.remove();
            }, 5000);
        }
    }

    showStep(stepNumber) {
        // Hide all steps
        this.step1.classList.remove('active');
        this.step2.classList.remove('active');
        this.step3.classList.remove('active');
        this.step4.classList.remove('active');

        // Update progress
        this.progress1.classList.remove('active', 'completed');
        this.progress2.classList.remove('active', 'completed');
        this.progress3.classList.remove('active', 'completed');

        // Show requested step and update progress
        switch(stepNumber) {
            case 1:
                this.step1.classList.add('active');
                this.progress1.classList.add('active');
                this.messageArea.innerHTML = '';
                break;
            case 2:
                this.step2.classList.add('active');
                this.progress1.classList.add('completed');
                this.progress2.classList.add('active');
                this.otpInput.focus();
                this.otpInput.value = '';
                this.messageArea.innerHTML = '';
                break;
            case 3:
                this.step3.classList.add('active');
                this.progress1.classList.add('completed');
                this.progress2.classList.add('completed');
                this.progress3.classList.add('active');
                this.newPinInput.focus();
                this.newPinInput.value = '';
                this.confirmPinInput.value = '';
                this.messageArea.innerHTML = '';
                break;
            case 4:
                this.step4.classList.add('active');
                this.progress1.classList.add('completed');
                this.progress2.classList.add('completed');
                this.progress3.classList.add('completed');
                break;
        }

        this.currentStep = stepNumber;
    }

    async requestOTP() {
        console.log('📧 Requesting OTP...');
        this.btnRequestOtp.disabled = true;
        this.btnRequestOtp.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

        try {
            const response = await fetch('/api/password/request-otp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            });

            const data = await response.json();

            if (response.ok) {
                this.email = data.email || this.email;
                console.log('✅ OTP sent successfully');
                this.showMessage('OTP sent to ' + this.email + ' successfully!', 'success');
                
                setTimeout(() => {
                    this.showStep(2);
                }, 1500);
            } else {
                console.error('❌ OTP request failed:', data);
                this.showMessage(data.message || 'Failed to send OTP. Please try again.', 'error');
            }
        } catch (error) {
            console.error('❌ Network error:', error);
            this.showMessage('Network error. Please check your connection.', 'error');
        } finally {
            this.btnRequestOtp.disabled = false;
            this.btnRequestOtp.innerHTML = '<i class="fas fa-paper-plane"></i> Send OTP';
        }
    }

    async verifyOTP() {
        const otp = this.otpInput.value.trim();

        // Validate OTP format
        if (!otp) {
            this.showMessage('Please enter the OTP code.', 'error');
            return;
        }

        if (!/^\d{6}$/.test(otp)) {
            this.showMessage('OTP must be exactly 6 digits.', 'error');
            return;
        }

        console.log('🔐 Verifying OTP...');
        this.btnVerifyOtp.disabled = true;
        this.btnVerifyOtp.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying...';

        try {
            const response = await fetch('/api/password/verify-otp', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ otp: otp })
            });

            const data = await response.json();

            if (response.ok) {
                this.otpVerified = true;
                console.log('✅ OTP verified successfully');
                this.showMessage('OTP verified successfully!', 'success');
                
                setTimeout(() => {
                    this.showStep(3);
                }, 1500);
            } else {
                console.error('❌ OTP verification failed:', data);
                this.showMessage(data.message || 'Invalid OTP. Please try again.', 'error');
            }
        } catch (error) {
            console.error('❌ Network error:', error);
            this.showMessage('Network error. Please try again.', 'error');
        } finally {
            this.btnVerifyOtp.disabled = false;
            this.btnVerifyOtp.innerHTML = '<i class="fas fa-check"></i> Verify OTP';
        }
    }

    async changePin() {
        const newPin = this.newPinInput.value.trim();
        const confirmPin = this.confirmPinInput.value.trim();

        // Validate new PIN
        if (!newPin) {
            this.showMessage('Please enter a new PIN.', 'error');
            return;
        }

        if (!/^\d{4,8}$/.test(newPin)) {
            this.showMessage('PIN must be 4-8 digits only.', 'error');
            return;
        }

        if (!confirmPin) {
            this.showMessage('Please confirm your PIN.', 'error');
            return;
        }

        if (newPin !== confirmPin) {
            this.showMessage('PINs do not match. Please try again.', 'error');
            return;
        }

        if (!this.otpVerified) {
            this.showMessage('OTP verification required. Please verify OTP first.', 'error');
            return;
        }

        console.log('🔐 Changing PIN...');
        this.btnChangePin.disabled = true;
        this.btnChangePin.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';

        try {
            const response = await fetch('/api/password/change', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    new_pin: newPin,
                    confirm_pin: confirmPin
                })
            });

            const data = await response.json();

            if (response.ok) {
                console.log('✅ PIN changed successfully');
                this.showMessage('PIN changed successfully!', 'success');
                
                setTimeout(() => {
                    this.showStep(4);
                }, 1500);
            } else {
                console.error('❌ PIN change failed:', data);
                this.showMessage(data.message || 'Failed to change PIN. Please try again.', 'error');
            }
        } catch (error) {
            console.error('❌ Network error:', error);
            this.showMessage('Network error. Please try again.', 'error');
        } finally {
            this.btnChangePin.disabled = false;
            this.btnChangePin.innerHTML = '<i class="fas fa-save"></i> Save New PIN';
        }
    }

    closeWindow() {
        // Close the window/tab
        window.close();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🔐 Password Change Page Loaded');
    new PasswordChangeForm();
});
