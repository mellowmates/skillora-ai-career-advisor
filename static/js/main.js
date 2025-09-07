// Skillora AI Career Advisor - Main JavaScript

// Global variables
let currentAssessmentStep = 0;
let assessmentData = {};
let chatHistory = [];

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Initialize page-specific functionality
    const currentPage = document.body.getAttribute('data-page');
    switch(currentPage) {
        case 'assessment':
            initializeAssessment();
            break;
        case 'chatbot':
            initializeChatbot();
            break;
        case 'dashboard':
            initializeDashboard();
            break;
    }
    
    // Add loading overlay functionality
    setupLoadingOverlay();
}

// Loading overlay functions
function showLoading(message = 'Loading...') {
    const overlay = document.getElementById('loading-overlay');
    const text = document.getElementById('loading-text');
    if (overlay) {
        if (text) text.textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function setupLoadingOverlay() {
    // Create loading overlay if it doesn't exist
    if (!document.getElementById('loading-overlay')) {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content">
                <div class="loading"></div>
                <div id="loading-text" class="loading-text">Loading...</div>
            </div>
        `;
        document.body.appendChild(overlay);
    }
}

// Assessment functionality
function initializeAssessment() {
    console.log('Initializing assessment...');
    
    // Handle form submissions
    const personalityForm = document.getElementById('personality-form');
    const skillsForm = document.getElementById('skills-form');
    const preferencesForm = document.getElementById('preferences-form');
    
    if (personalityForm) {
        personalityForm.addEventListener('submit', handlePersonalitySubmission);
    }
    
    if (skillsForm) {
        skillsForm.addEventListener('submit', handleSkillsSubmission);
    }
    
    if (preferencesForm) {
        preferencesForm.addEventListener('submit', handlePreferencesSubmission);
    }
    
    // Initialize assessment progress tracking
    updateAssessmentProgress();
}

// Handle personality assessment submission
async function handlePersonalitySubmission(event) {
    event.preventDefault();
    showLoading('Analyzing personality...');
    
    try {
        const formData = new FormData(event.target);
        const responses = {};
        
        // Process personality responses
        for (let [key, value] of formData.entries()) {
            const [trait, questionNum] = key.split('_');
            if (!responses[trait]) responses[trait] = [];
            responses[trait].push(parseInt(value));
        }
        
        const response = await fetch('/api/personality/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ responses })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Mark as completed in session storage
            sessionStorage.setItem('personality_completed', 'true');
            showSuccess('Personality assessment completed!');
            // Move to next section
            switchToAssessmentSection('skills');
            updateAssessmentProgress();
        } else {
            showError('Failed to process personality assessment');
        }
    } catch (error) {
        console.error('Error submitting personality assessment:', error);
        showError('Failed to submit personality assessment');
    } finally {
        hideLoading();
    }
}

// Handle skills assessment submission
async function handleSkillsSubmission(event) {
    event.preventDefault();
    showLoading('Analyzing skills...');
    
    try {
        const formData = new FormData(event.target);
        const skillsData = {
            technical: parseSkillsList(formData.get('technical_skills')),
            soft: parseSkillsList(formData.get('soft_skills')),
            domain: parseSkillsList(formData.get('domain_skills'))
        };
        
        const response = await fetch('/api/skills/assess', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                skills: skillsData,
                experience: '',
                education: ''
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Mark as completed in session storage
            sessionStorage.setItem('skills_completed', 'true');
            showSuccess('Skills assessment completed!');
            // Move to next section
            switchToAssessmentSection('preferences');
            updateAssessmentProgress();
        } else {
            showError('Failed to process skills assessment');
        }
    } catch (error) {
        console.error('Error submitting skills assessment:', error);
        showError('Failed to submit skills assessment');
    } finally {
        hideLoading();
    }
}

// Handle preferences submission
async function handlePreferencesSubmission(event) {
    event.preventDefault();
    showLoading('Processing preferences...');
    
    try {
        const formData = new FormData(event.target);
        const preferences = {
            work_environment: Array.from(formData.getAll('work_env')),
            work_style: formData.get('work_style'),
            priority: formData.get('priority')
        };
        
        // Get career recommendations
        const response = await fetch('/api/career/match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                preferences,
                user_data: {}
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Mark as completed in session storage
            sessionStorage.setItem('preferences_completed', 'true');
            showSuccess('Assessment completed! Generating recommendations...');
            // Show results section
            showAssessmentResults();
        } else {
            showError('Failed to generate career recommendations');
        }
    } catch (error) {
        console.error('Error submitting preferences:', error);
        showError('Failed to complete assessment');
    } finally {
        hideLoading();
    }
}

// Utility functions for assessment
function parseSkillsList(skillsText) {
    if (!skillsText) return [];
    
    return skillsText.split(',').map(s => s.trim()).filter(s => s.length > 0);
}

function switchToAssessmentSection(sectionName) {
    // Remove active class from all sections and nav buttons
    document.querySelectorAll('.assessment-content').forEach(section => {
        section.classList.remove('active');
    });
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Activate the target section
    const targetSection = document.getElementById(sectionName + '-section');
    const targetBtn = document.querySelector(`[data-section="${sectionName}"]`);
    
    if (targetSection) targetSection.classList.add('active');
    if (targetBtn) targetBtn.classList.add('active');
}

function updateAssessmentProgress() {
    const progressBar = document.getElementById('overall-progress');
    if (!progressBar) return;
    
    // Calculate progress based on completed sections
    let completedSections = 0;
    const totalSections = 3;
    
    // Check if sections are completed (this is a simple implementation)
    // In a real app, you'd track this more accurately
    if (sessionStorage.getItem('personality_completed')) completedSections++;
    if (sessionStorage.getItem('skills_completed')) completedSections++;
    if (sessionStorage.getItem('preferences_completed')) completedSections++;
    
    const progress = (completedSections / totalSections) * 100;
    progressBar.style.width = progress + '%';
}

function showAssessmentResults() {
    // Hide all assessment sections
    document.querySelectorAll('.assessment-content').forEach(section => {
        section.style.display = 'none';
    });
    
    // Show results section
    const resultsSection = document.getElementById('assessment-results');
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
    
    // Update progress to 100%
    const progressBar = document.getElementById('overall-progress');
    if (progressBar) {
        progressBar.style.width = '100%';
    }
}

// Skip assessment section
function skipAssessmentSection(sectionType) {
    console.log(`Skipping ${sectionType} assessment`);
    
    // Mark as completed in session storage
    sessionStorage.setItem(`${sectionType}_completed`, 'skipped');
    
    // Move to next section
    if (sectionType === 'personality') {
        switchToAssessmentSection('skills');
    } else if (sectionType === 'skills') {
        switchToAssessmentSection('preferences');
    } else if (sectionType === 'preferences') {
        showAssessmentResults();
    }
    
    updateAssessmentProgress();
    showSuccess(`${sectionType.charAt(0).toUpperCase() + sectionType.slice(1)} assessment skipped`);
}

// Chatbot functionality
function initializeChatbot() {
    console.log('Initializing chatbot...');
    
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-message');
    const chatMessages = document.getElementById('chat-messages');
    
    if (chatInput && sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    // Load chat history
    loadChatHistory();
}

async function sendMessage() {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    // Disable send button during processing
    if (sendBtn) sendBtn.disabled = true;
    
    // Add user message to chat
    addMessageToChat('user', message);
    chatInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch('/api/chatbot/message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_id: 'default'
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            addMessageToChat('bot', data.data.response);
        } else {
            addMessageToChat('bot', 'Sorry, I encountered an error. Please try again.');
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToChat('bot', 'Sorry, the chatbot is currently unavailable. Please try again later.');
    } finally {
        hideTypingIndicator();
        // Re-enable send button
        if (sendBtn) sendBtn.disabled = false;
    }
}

function addMessageToChat(sender, message) {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    if (sender === 'bot') {
        messageDiv.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="message-content">
                ${message}
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="avatar">👤</div>
            <div class="message-content">
                ${message}
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Store in chat history
    chatHistory.push({ sender, message, timestamp: new Date() });
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chat-messages');
    if (!chatMessages) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'message bot-message typing';
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function loadChatHistory() {
    // This would load chat history from session storage or server
    console.log('Loading chat history...');
}

// Dashboard functionality
function initializeDashboard() {
    console.log('Initializing dashboard...');
    
    // Load user data and recommendations
    loadDashboardData();
}

async function loadDashboardData() {
    // This would load user's career recommendations, progress, etc.
    console.log('Loading dashboard data...');
}

// Utility functions
function showError(message) {
    // Create or show error toast/alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-circle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function showSuccess(message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show position-fixed';
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
    alertDiv.innerHTML = `
        <i class="fas fa-check-circle me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// API helper functions
async function apiCall(endpoint, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const finalOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(endpoint, finalOptions);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Export functions for use in other scripts
window.Skillora = {
    showLoading,
    hideLoading,
    showError,
    showSuccess,
    apiCall,
    sendMessage,
    handlePersonalitySubmission,
    handleSkillsSubmission,
    handlePreferencesSubmission,
    skipAssessmentSection
};

// Make functions available globally for templates
window.app = {
    skipAssessmentSection,
    sendChatMessage: sendMessage  // Add alias for chatbot template
};