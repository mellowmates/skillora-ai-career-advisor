# Skillora AI Career Advisor

A comprehensive AI-powered career guidance platform that analyzes personality traits, skills, and market trends to provide personalized career recommendations.

## 🚀 Features

- **Intelligent Personality Analysis**: Advanced rule-based system to analyze personality traits using the Big Five model
- **Comprehensive Skills Assessment**: Evaluates technical, soft, and domain-specific skills
- **Real-Time Job Market Data**: Scrapes current job openings and market trends
- **Career Matching Algorithm**: Advanced matching system combining personality, skills, and preferences
- **Career Roadmapping**: Detailed career progression paths and skill development plans
- **Interactive Web Interface**: Modern, responsive design with smooth user experience

## 🏗️ Project Structure

```
skillora-ai-career-advisor/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                          # Main Flask application
├── modules/                        # Core application modules
│   ├── __init__.py
│   ├── personality_analyzer.py     # AI-powered personality assessment
│   ├── skills_assessor.py         # Skills evaluation and analysis
│   ├── job_market_scraper.py      # Job market data collection
│   └── career_matcher.py          # Career matching algorithm
├── data/                          # Data files
│   ├── career_profiles.json       # Career information database
│   ├── skills_mapping.json        # Skills-to-career mapping
│   └── job_market_data.json       # Market trends and statistics
├── templates/                     # HTML templates
│   └── index.html                 # Main application interface
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css              # Custom styles
│   └── js/
│       └── app.js                 # Frontend JavaScript
└── docs/                          # Documentation
    └── project_plan.md            # Comprehensive project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- No external API keys required!

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skillora-ai-career-advisor.git
   cd skillora-ai-career-advisor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## 🎯 Usage

### Assessment Process

1. **Personality Assessment**: Answer 8 questions about your personality traits and work preferences
2. **Skills Evaluation**: Rate your skills and provide information about your experience and education
3. **Career Preferences**: Specify your preferred work environment, salary expectations, and location
4. **Get Results**: Receive personalized career recommendations with detailed analysis

### Key Features

- **Personality Analysis**: Get insights into your Big Five personality traits
- **Skills Profile**: Comprehensive evaluation of your technical and soft skills
- **Career Matches**: AI-powered career recommendations with compatibility scores
- **Market Intelligence**: Current job market trends and salary information
- **Career Roadmaps**: Detailed progression paths for your top career matches

## 🔧 API Endpoints

### Assessment Endpoints
- `GET /api/personality/questions` - Get personality assessment questions
- `POST /api/personality/analyze` - Analyze personality responses
- `GET /api/skills/questions` - Get skills assessment questions
- `POST /api/skills/assess` - Assess user skills
- `POST /api/career/match` - Match user with careers

### Data Endpoints
- `GET /api/careers/list` - Get available careers
- `GET /api/skills/mapping` - Get skills mapping data
- `GET /api/jobs/market-data` - Get job market data
- `POST /api/jobs/recommendations` - Get job recommendations

### Utility Endpoints
- `GET /api/health` - Health check
- `POST /api/session/clear` - Clear session data
- `GET /api/report/download` - Download assessment report

## 🧠 AI and Machine Learning

### Personality Analysis
- Advanced rule-based system for personality assessment
- Implements Big Five personality model (OCEAN)
- Provides detailed personality insights and work style analysis
- No external API dependencies required

### Skills Assessment
- Multi-source skills evaluation (self-assessment, experience, education)
- Skill categorization and level assessment
- Gap analysis and development recommendations

### Career Matching
- Advanced compatibility scoring algorithm
- Combines personality, skills, and preferences
- Market-aligned career recommendations

## 📊 Data Sources

### Career Profiles
- Comprehensive career information database
- Salary ranges and growth outlook
- Required and preferred skills
- Education requirements and career paths

### Skills Mapping
- Skills-to-career relevance mapping
- Learning resources and certifications
- Career relevance scores

### Job Market Data
- Real-time job market trends
- Salary benchmarks and growth rates
- Skill demand analysis
- Geographic market insights

## 🔒 Privacy and Security

- **No Permanent Storage**: User data is only stored in temporary sessions
- **Data Privacy**: Clear data usage policies and GDPR compliance considerations
- **Secure API**: Input validation, rate limiting, and secure session management
- **External Integrations**: Secure API key management and fallback mechanisms

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Bootstrap for the responsive UI framework
- Font Awesome for the icon library
- The open-source community for various Python packages
- Big Five personality model research and validation

## 📞 Support

For support, email support@skillora.com or create an issue in the GitHub repository.

## 🔮 Future Enhancements

- User accounts and profile saving
- Advanced career analytics
- Industry-specific assessments
- Mobile application
- AI-powered resume optimization
- Interview preparation tools
- Corporate partnerships and enterprise solutions

---

**Built with ❤️ by the Skillora Team**
AI-powered personalized career advisor with real-time job market intelligence - Gen AI Hackathon Submission