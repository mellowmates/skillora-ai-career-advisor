// Skillora AI Career Advisor - Main JavaScript Application

class CareerAdvisorApp {
    constructor() {
        this.currentStep = 0;
        this.totalSteps = 3;
        this.assessmentData = {
            personality: {},
            skills: {},
            preferences: {}
        };
        this.results = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadPersonalityQuestions();
    }

    setupEventListeners() {
        // Navigation smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            });
        });

        // Form validation
        document.addEventListener('input', (e) => {
            if (e.target.matches('input, select, textarea')) {
                this.validateForm();
            }
        });
    }

    async loadPersonalityQuestions() {
        try {
            const response = await fetch('/api/personality/questions');
            const data = await response.json();
            
            if (data.success) {
                this.renderPersonalityQuestions(data.questions);
            } else {
                this.showError('Failed to load personality questions');
            }
        } catch (error) {
            this.showError('Error loading personality questions: ' + error.message);
        }
    }

    renderPersonalityQuestions(questions) {
        const container = document.getElementById('personalityQuestions');
        container.innerHTML = '';

        questions.forEach((question, index) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'question-item';
            questionDiv.innerHTML = `
                <div class="question-text">${question.question}</div>
                <div class="option-group">
                    ${this.renderLikertScale(question.id, question.scale)}
                </div>
            `;
            container.appendChild(questionDiv);
        });
    }

    renderLikertScale(questionId, scale) {
        const labels = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'];
        let html = '';
        
        for (let i = 1; i <= scale; i++) {
            html += `
                <div class="option-item">
                    <input type="radio" name="${questionId}" value="${i}" id="${questionId}_${i}">
                    <label for="${questionId}_${i}">${labels[i-1]}</label>
                </div>
            `;
        }
        
        return html;
    }

    async submitPersonalityAssessment() {
        const responses = this.collectPersonalityResponses();
        
        if (Object.keys(responses).length === 0) {
            this.showError('Please answer all personality questions');
            return;
        }

        try {
            this.showLoading('Analyzing your personality...');
            
            const response = await fetch('/api/personality/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ responses })
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.assessmentData.personality = data.personality_profile;
                this.nextStep();
                this.loadSkillsQuestions();
            } else {
                this.showError(data.error || 'Failed to analyze personality');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Error analyzing personality: ' + error.message);
        }
    }

    collectPersonalityResponses() {
        const responses = {};
        const inputs = document.querySelectorAll('#personalityQuestions input[type="radio"]:checked');
        
        inputs.forEach(input => {
            responses[input.name] = parseInt(input.value);
        });
        
        return responses;
    }

    async loadSkillsQuestions() {
        try {
            const response = await fetch('/api/skills/questions');
            const data = await response.json();
            
            if (data.success) {
                this.renderSkillsQuestions(data.questions);
            } else {
                this.showError('Failed to load skills questions');
            }
        } catch (error) {
            this.showError('Error loading skills questions: ' + error.message);
        }
    }

    renderSkillsQuestions(questions) {
        const container = document.getElementById('skillsQuestions');
        container.innerHTML = '';

        questions.forEach((question, index) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'question-item';
            questionDiv.innerHTML = `
                <div class="question-text">${question.question}</div>
                <div class="option-group">
                    ${this.renderSkillLevelOptions(question.id, question.options)}
                </div>
            `;
            container.appendChild(questionDiv);
        });

        // Add additional skills input
        const additionalSkillsDiv = document.createElement('div');
        additionalSkillsDiv.className = 'question-item';
        additionalSkillsDiv.innerHTML = `
            <div class="question-text">Additional Skills (Optional)</div>
            <div class="mb-3">
                <label class="form-label">Work Experience</label>
                <textarea class="form-control" id="workExperience" rows="3" 
                    placeholder="Describe your work experience and the skills you've developed..."></textarea>
            </div>
            <div class="mb-3">
                <label class="form-label">Education</label>
                <textarea class="form-control" id="education" rows="2" 
                    placeholder="Describe your educational background..."></textarea>
            </div>
            <div class="mb-3">
                <label class="form-label">Certifications</label>
                <textarea class="form-control" id="certifications" rows="2" 
                    placeholder="List any certifications or courses you've completed..."></textarea>
            </div>
        `;
        container.appendChild(additionalSkillsDiv);
    }

    renderSkillLevelOptions(questionId, options) {
        let html = '';
        
        options.forEach((option, index) => {
            html += `
                <div class="option-item">
                    <input type="radio" name="${questionId}" value="${option.toLowerCase()}" id="${questionId}_${index}">
                    <label for="${questionId}_${index}">${option}</label>
                </div>
            `;
        });
        
        return html;
    }

    async submitSkillsAssessment() {
        const userInput = this.collectSkillsData();
        
        if (Object.keys(userInput.self_assessed_skills || {}).length === 0) {
            this.showError('Please rate at least some of your skills');
            return;
        }

        try {
            this.showLoading('Assessing your skills...');
            
            const response = await fetch('/api/skills/assess', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_input: userInput })
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.assessmentData.skills = data.skills_profile;
                this.nextStep();
            } else {
                this.showError(data.error || 'Failed to assess skills');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Error assessing skills: ' + error.message);
        }
    }

    collectSkillsData() {
        const selfAssessedSkills = {};
        const inputs = document.querySelectorAll('#skillsQuestions input[type="radio"]:checked');
        
        inputs.forEach(input => {
            selfAssessedSkills[input.name] = input.value;
        });

        return {
            self_assessed_skills: selfAssessedSkills,
            work_experience: document.getElementById('workExperience')?.value || '',
            education: document.getElementById('education')?.value || '',
            certifications: document.getElementById('certifications')?.value || ''
        };
    }

    async submitPreferences() {
        const preferences = this.collectPreferences();
        
        try {
            this.showLoading('Finding your perfect career matches...');
            
            const response = await fetch('/api/career/match', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ preferences })
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.results = data;
                this.showResults();
            } else {
                this.showError(data.error || 'Failed to find career matches');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Error finding career matches: ' + error.message);
        }
    }

    collectPreferences() {
        return {
            work_environment: document.getElementById('workEnvironment').value,
            salary_expectation: parseInt(document.getElementById('salaryExpectation').value) || 0,
            work_life_balance: document.getElementById('workLifeBalance').value,
            location: document.getElementById('preferredLocation').value
        };
    }

    showResults() {
        // Hide assessment section
        document.getElementById('assessment').style.display = 'none';
        
        // Show results section
        document.getElementById('results').style.display = 'block';
        document.getElementById('resultsContent').classList.remove('d-none');
        
        // Scroll to results
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        
        // Render results
        this.renderCareerMatches();
        this.renderSkillsAnalysis();
        this.renderPersonalityInsights();
        this.renderJobMarketInsights();
    }

    renderCareerMatches() {
        const container = document.getElementById('careerMatches');
        container.innerHTML = '';

        if (!this.results.top_careers || this.results.top_careers.length === 0) {
            container.innerHTML = '<p class="text-muted">No career matches found.</p>';
            return;
        }

        this.results.top_careers.slice(0, 5).forEach((career, index) => {
            const careerCard = document.createElement('div');
            careerCard.className = 'career-match-card fade-in';
            careerCard.style.animationDelay = `${index * 0.1}s`;
            
            careerCard.innerHTML = `
                <div class="career-match-content">
                    <div class="match-score">
                        ${Math.round(career.total_score * 100)}% Match
                    </div>
                    <h4 class="mb-3">${career.title}</h4>
                    <p class="mb-3">${career.description}</p>
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Growth Outlook:</strong> ${career.growth_outlook}</p>
                            <p><strong>Work Environment:</strong> ${career.work_environment}</p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Salary Range:</strong> $${career.salary_range.min?.toLocaleString()} - $${career.salary_range.max?.toLocaleString()}</p>
                            <p><strong>Education:</strong> ${career.education_requirements}</p>
                        </div>
                    </div>
                    <div class="mt-3">
                        <button class="btn btn-light me-2" onclick="app.getCareerRoadmap('${career.career_id}')">
                            <i class="fas fa-route me-1"></i>View Roadmap
                        </button>
                        <button class="btn btn-outline-light" onclick="app.getJobRecommendations('${career.title}')">
                            <i class="fas fa-briefcase me-1"></i>Find Jobs
                        </button>
                    </div>
                </div>
            `;
            
            container.appendChild(careerCard);
        });
    }

    renderSkillsAnalysis() {
        const container = document.getElementById('skillsAnalysis');
        const skillsProfile = this.assessmentData.skills;
        
        if (!skillsProfile || !skillsProfile.skills) {
            container.innerHTML = '<p class="text-muted">No skills data available.</p>';
            return;
        }

        let html = `
            <div class="mb-3">
                <h6>Overall Skills Score: ${Math.round(skillsProfile.overall_score || 0)}/100</h6>
                <div class="progress">
                    <div class="progress-bar" style="width: ${skillsProfile.overall_score || 0}%"></div>
                </div>
            </div>
        `;

        // Show skill categories
        if (skillsProfile.categories) {
            Object.entries(skillsProfile.categories).forEach(([category, skills]) => {
                if (skills.length > 0) {
                    html += `
                        <div class="mb-3">
                            <h6 class="text-capitalize">${category} Skills</h6>
                            ${skills.map(skill => `<span class="skill-badge">${skill}</span>`).join('')}
                        </div>
                    `;
                }
            });
        }

        // Show recommendations
        if (skillsProfile.recommendations && skillsProfile.recommendations.length > 0) {
            html += `
                <div class="mt-3">
                    <h6>Recommendations</h6>
                    <ul class="list-unstyled">
                        ${skillsProfile.recommendations.map(rec => `<li><i class="fas fa-arrow-right text-primary me-2"></i>${rec}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    renderPersonalityInsights() {
        const container = document.getElementById('personalityInsights');
        const personalityProfile = this.assessmentData.personality;
        
        if (!personalityProfile || !personalityProfile.big_five_scores) {
            container.innerHTML = '<p class="text-muted">No personality data available.</p>';
            return;
        }

        let html = '<h6>Big Five Personality Traits</h6>';
        
        Object.entries(personalityProfile.big_five_scores).forEach(([trait, score]) => {
            html += `
                <div class="personality-trait">
                    <span class="text-capitalize">${trait}</span>
                    <div class="trait-bar">
                        <div class="trait-fill" style="width: ${score}%"></div>
                    </div>
                    <span class="ms-2">${score}/100</span>
                </div>
            `;
        });

        // Show characteristics
        if (personalityProfile.characteristics && personalityProfile.characteristics.length > 0) {
            html += `
                <div class="mt-3">
                    <h6>Key Characteristics</h6>
                    ${personalityProfile.characteristics.map(char => `<span class="skill-badge">${char}</span>`).join('')}
                </div>
            `;
        }

        container.innerHTML = html;
    }

    async renderJobMarketInsights() {
        const container = document.getElementById('jobMarketInsights');
        
        try {
            const response = await fetch('/api/jobs/market-data');
            const data = await response.json();
            
            if (data.success) {
                const marketData = data.market_data;
                
                let html = `
                    <div class="row">
                        <div class="col-md-6">
                            <div class="market-insight">
                                <div class="insight-title">Remote Work Trend</div>
                                <p>${marketData.remote_work_analysis.fully_remote_percentage}% of jobs are fully remote</p>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="market-insight">
                                <div class="insight-title">Job Growth Rate</div>
                                <p>Overall job market growing at ${marketData.market_trends.job_growth_rate}% annually</p>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-12">
                            <h6>High Demand Skills</h6>
                            <div class="mb-3">
                                ${Object.entries(marketData.market_trends.skill_demand_trends)
                                    .filter(([skill, demand]) => demand === 'high')
                                    .map(([skill, demand]) => `<span class="skill-badge">${skill.replace('_', ' ')}</span>`)
                                    .join('')}
                            </div>
                        </div>
                    </div>
                `;
                
                container.innerHTML = html;
            } else {
                container.innerHTML = '<p class="text-muted">Unable to load market data.</p>';
            }
        } catch (error) {
            container.innerHTML = '<p class="text-muted">Error loading market data.</p>';
        }
    }

    async getCareerRoadmap(careerId) {
        try {
            const response = await fetch(`/api/career/roadmap/${careerId}`);
            const data = await response.json();
            
            if (data.success) {
                this.showRoadmapModal(data.roadmap);
            } else {
                this.showError('Failed to load career roadmap');
            }
        } catch (error) {
            this.showError('Error loading career roadmap: ' + error.message);
        }
    }

    showRoadmapModal(roadmap) {
        // Create and show modal with roadmap data
        const modalHtml = `
            <div class="modal fade" id="roadmapModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">${roadmap.career_title} Career Roadmap</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Entry Level</h6>
                                    <ul>
                                        ${roadmap.roadmap.entry_level.map(role => `<li>${role}</li>`).join('')}
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6>Mid Level</h6>
                                    <ul>
                                        ${roadmap.roadmap.mid_level.map(role => `<li>${role}</li>`).join('')}
                                    </ul>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Senior Level</h6>
                                    <ul>
                                        ${roadmap.roadmap.senior_level.map(role => `<li>${role}</li>`).join('')}
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6>Leadership</h6>
                                    <ul>
                                        ${roadmap.roadmap.leadership.map(role => `<li>${role}</li>`).join('')}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if any
        const existingModal = document.getElementById('roadmapModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Add new modal
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('roadmapModal'));
        modal.show();
    }

    async getJobRecommendations(careerTitle) {
        try {
            this.showLoading('Finding job opportunities...');
            
            const response = await fetch('/api/jobs/recommendations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    location: document.getElementById('preferredLocation').value || '',
                    career_title: careerTitle
                })
            });

            const data = await response.json();
            this.hideLoading();

            if (data.success) {
                this.showJobRecommendationsModal(data.recommendations);
            } else {
                this.showError('Failed to find job recommendations');
            }
        } catch (error) {
            this.hideLoading();
            this.showError('Error finding job recommendations: ' + error.message);
        }
    }

    showJobRecommendationsModal(jobs) {
        const modalHtml = `
            <div class="modal fade" id="jobsModal" tabindex="-1">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Job Recommendations</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                ${jobs.slice(0, 10).map(job => `
                                    <div class="col-md-6 mb-3">
                                        <div class="card">
                                            <div class="card-body">
                                                <h6 class="card-title">${job.title}</h6>
                                                <p class="card-text">
                                                    <strong>Company:</strong> ${job.company}<br>
                                                    <strong>Location:</strong> ${job.location}<br>
                                                    <strong>Salary:</strong> ${job.salary}
                                                </p>
                                                <a href="${job.url}" target="_blank" class="btn btn-primary btn-sm">View Job</a>
                                            </div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Remove existing modal if any
        const existingModal = document.getElementById('jobsModal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Add new modal
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('jobsModal'));
        modal.show();
    }

    async downloadReport() {
        try {
            const response = await fetch('/api/report/download');
            const data = await response.json();
            
            if (data.success) {
                // Create and download report
                const report = data.report;
                const reportText = this.formatReportForDownload(report);
                
                const blob = new Blob([reportText], { type: 'text/plain' });
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `career-assessment-report-${new Date().toISOString().split('T')[0]}.txt`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            } else {
                this.showError('Failed to download report');
            }
        } catch (error) {
            this.showError('Error downloading report: ' + error.message);
        }
    }

    formatReportForDownload(report) {
        let text = 'SKILLORA AI CAREER ADVISOR - ASSESSMENT REPORT\n';
        text += '=' .repeat(50) + '\n\n';
        text += `Generated: ${report.timestamp}\n\n`;
        
        text += 'PERSONALITY PROFILE\n';
        text += '-'.repeat(20) + '\n';
        if (report.personality_profile) {
            const traits = report.personality_profile.big_five_scores || {};
            Object.entries(traits).forEach(([trait, score]) => {
                text += `${trait}: ${score}/100\n`;
            });
        }
        
        text += '\nSKILLS PROFILE\n';
        text += '-'.repeat(15) + '\n';
        if (report.skills_profile) {
            text += `Overall Score: ${Math.round(report.skills_profile.overall_score || 0)}/100\n`;
        }
        
        text += '\nTOP CAREER MATCHES\n';
        text += '-'.repeat(20) + '\n';
        if (report.career_matches) {
            report.career_matches.slice(0, 5).forEach((career, index) => {
                text += `${index + 1}. ${career.title} (${Math.round(career.total_score * 100)}% match)\n`;
            });
        }
        
        text += '\nRECOMMENDATIONS\n';
        text += '-'.repeat(15) + '\n';
        if (report.recommendations) {
            report.recommendations.next_steps.forEach(step => {
                text += `• ${step}\n`;
            });
        }
        
        return text;
    }

    nextStep() {
        this.currentStep++;
        this.updateProgress();
        
        // Hide current step
        const currentStepElement = document.querySelector('.assessment-step:not(.d-none)');
        if (currentStepElement) {
            currentStepElement.classList.add('d-none');
        }
        
        // Show next step
        const stepIds = ['personalityStep', 'skillsStep', 'preferencesStep'];
        if (this.currentStep < stepIds.length) {
            document.getElementById(stepIds[this.currentStep]).classList.remove('d-none');
        }
    }

    updateProgress() {
        const progress = (this.currentStep / this.totalSteps) * 100;
        document.getElementById('progressBar').style.width = `${progress}%`;
    }

    showLoading(message = 'Loading...') {
        const loadingDiv = document.getElementById('loadingResults');
        if (loadingDiv) {
            loadingDiv.classList.remove('d-none');
            loadingDiv.querySelector('p').textContent = message;
        }
    }

    hideLoading() {
        const loadingDiv = document.getElementById('loadingResults');
        if (loadingDiv) {
            loadingDiv.classList.add('d-none');
        }
    }

    showError(message) {
        // Create and show error alert
        const alertHtml = `
            <div class="alert alert-danger alert-dismissible fade show" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Insert at top of assessment container
        const container = document.getElementById('assessmentContainer');
        container.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }

    validateForm() {
        // Basic form validation logic
        const requiredFields = document.querySelectorAll('[required]');
        let isValid = true;
        
        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                isValid = false;
                field.classList.add('is-invalid');
            } else {
                field.classList.remove('is-invalid');
            }
        });
        
        return isValid;
    }

    startNewAssessment() {
        // Reset application state
        this.currentStep = 0;
        this.assessmentData = { personality: {}, skills: {}, preferences: {} };
        this.results = null;
        
        // Reset UI
        document.getElementById('assessment').style.display = 'block';
        document.getElementById('results').style.display = 'none';
        document.getElementById('resultsContent').classList.add('d-none');
        
        // Reset progress
        this.updateProgress();
        
        // Show first step
        document.querySelectorAll('.assessment-step').forEach(step => {
            step.classList.add('d-none');
        });
        document.getElementById('personalityStep').classList.remove('d-none');
        
        // Clear forms
        document.querySelectorAll('input, select, textarea').forEach(field => {
            field.value = '';
            field.classList.remove('is-invalid');
        });
        
        // Scroll to assessment
        document.getElementById('assessment').scrollIntoView({ behavior: 'smooth' });
    }
}

// Global functions for HTML onclick handlers
function startAssessment() {
    document.getElementById('assessment').scrollIntoView({ behavior: 'smooth' });
}

function scrollToSection(sectionId) {
    document.getElementById(sectionId).scrollIntoView({ behavior: 'smooth' });
}

function submitPersonalityAssessment() {
    app.submitPersonalityAssessment();
}

function submitSkillsAssessment() {
    app.submitSkillsAssessment();
}

function submitPreferences() {
    app.submitPreferences();
}

function downloadReport() {
    app.downloadReport();
}

function startNewAssessment() {
    app.startNewAssessment();
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new CareerAdvisorApp();
});
