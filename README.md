# Skillora AI Career Advisor

**GenAI Exchange Hackathon Submission - Team: The Algorithmic Alchemists**

A comprehensive AI-powered career guidance platform that leverages machine learning to analyze personality traits, skills, and market trends to provide personalized career recommendations for Indian students navigating the evolving job market.

## 🎯 Problem Statement

**Personalized Career and Skills Advisor**

Students in India often face a bewildering array of career choices, compounded by generic guidance that fails to account for their unique interests, aptitudes, and the rapidly evolving job market. The traditional approach to career counseling struggles to keep pace with the emergence of new job roles and the specific skills required to succeed in them.

## 🚀 Our Solution

Skillora AI leverages machine learning and data analytics to create a personalized career advisor that:

- **Intelligent Personality Analysis**: Advanced ML-based system analyzing personality traits using the Big Five model
- **Comprehensive Skills Assessment**: AI-powered evaluation of technical, soft, and domain-specific skills
- **Real-Time Market Intelligence**: Dynamic job market analysis with salary predictions and trend forecasting
- **Smart Career Matching**: ML algorithms combining personality, skills, preferences, and market data
- **Personalized Learning Roadmaps**: AI-generated career progression paths with actionable skill development plans
- **Interactive Web Platform**: Modern, responsive interface with real-time recommendations

## 🏗️ Project Structure

```
skillora-ai-career-advisor/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── app.py                          # Main Flask application
├── config.py                      # Configuration settings
├── run.py                         # Application runner
├── train_all_models.py            # ML model training pipeline
├── modules/                        # Core business logic modules
│   ├── __init__.py
│   ├── data_loader.py             # Data loading and management
│   ├── user_profiler.py           # User profile management
│   ├── personality_assessor.py    # Big Five personality analysis
│   ├── skills_assessor.py         # Skills evaluation and gap analysis
│   ├── career_matcher.py          # AI-powered career matching
│   ├── learning_roadmap.py        # Personalized learning path generation
│   └── market_analyzer.py         # Job market intelligence
├── models/                         # Machine Learning models
│   ├── data_collection/
│   │   ├── data_preprocessor.py   # Data preprocessing pipeline
│   │   ├── kaggle_datasets.py     # Dataset management
│   │   └── indian_job_scraper.py  # Job market data scraper
│   ├── training/                  # Model training scripts
│   │   ├── model_trainer.py       # Main training orchestrator
│   │   ├── career_recommendation_model.py  # Career model class
│   │   ├── salary_prediction_model.py      # Salary model class
│   │   ├── skills_prediction_model.py      # Skills model class
│   │   ├── train_career_model.py  # Career model training script
│   │   ├── train_salary_model.py  # Salary model training script
│   │   └── train_skills_model.py  # Skills model training script

│   └── model_evaluation.py        # Model evaluation utilities
├── data/                          # Data files
│   ├── career_profiles.json       # Career information database
│   ├── skills_mapping.json        # Skills-to-career mapping
│   ├── job_market_data.json       # Market trends and statistics
│   ├── personality_traits.json    # Personality assessment data
│   ├── learning_resources.json    # Learning resources database
│   ├── education_paths.json       # Education pathway data
│   ├── india_locations.json       # Indian cities and locations
│   ├── salary_data.json          # Salary benchmarks
│   └── chatbot_knowledge.json     # Chatbot knowledge base
├── templates/                     # HTML templates
│   ├── base.html                  # Base template
│   ├── index.html                 # Landing page
│   ├── about.html                 # About page
│   ├── profile.html               # User profile creation
│   ├── assessment.html            # Assessment interface
│   ├── dashboard.html             # Main dashboard
│   ├── career_detail.html         # Career details page
│   ├── roadmap.html               # Learning roadmap
│   └── chatbot.html               # AI chatbot interface
├── static/                        # Static assets
│   ├── css/
│   │   └── style.css              # Custom styles
│   ├── js/
│   │   └── app.js                 # Frontend JavaScript
│   └── images/                    # Image assets
├── chatbot/                       # Chatbot module
│   ├── __init__.py
│   ├── chatbot_api.py            # Chatbot API interface
│   ├── dialog_flow.py            # Dialog flow management
│   └── fallback_adapter.py       # Fallback response handler
├── utils/                         # Utility functions
│   └── __init__.py
└── docs/                          # Documentation
    ├── project_plan.md            # Comprehensive project documentation
    ├── api_documentation.md       # API documentation
    ├── chatbot_design.md          # Chatbot design documentation
    └── model_data_sources.md      # Data sources documentation
```

**Note**: The following directories are excluded from version control (see `.gitignore`):
- `data/raw/` - Large CSV datasets (generated during setup)
- `data/processed/` - Processed training data (generated during preprocessing)
- `models/trained_models/` - Trained ML models (.pkl files, generated during training)

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- 4GB+ RAM (for ML model training)
- No external API keys required!

### Quick Start

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

4. **Train ML Models (First Time Setup)**
   ```bash
   python train_all_models.py
   ```
   This will:
   - Create `data/raw/` and `data/processed/` directories
   - Process the raw career and salary datasets
   - Train career recommendation, salary prediction, and skills models
   - Create `models/trained_models/` directory with trained models (.pkl files)

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

### Development Setup

For development with hot reload:
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

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

## 🧠 AI and Machine Learning Architecture

### 1. Career Recommendation Model
- **Algorithm**: Random Forest Classifier
- **Features**: Personality traits, skills, experience, education
- **Output**: Top career matches with confidence scores
- **Performance**: Trained on 525+ career profiles

### 2. Salary Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Experience, location, skills, industry, company size
- **Output**: Salary range predictions with growth potential
- **Accuracy**: R² Score of 0.646, MAE of ₹3.97L

### 3. Skills Prediction Model
- **Algorithm**: Multi-output Random Forest
- **Features**: Career path, current role, experience, industry
- **Output**: Required skills for target careers
- **Capability**: Predicts 40+ different skill categories

### 4. Personality Analysis Engine
- **Model**: Big Five (OCEAN) personality framework
- **Method**: Weighted scoring algorithm
- **Features**: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- **Integration**: Feeds into career matching algorithm

### 5. Market Intelligence System
- **Data Sources**: Real job market data, salary surveys
- **Analysis**: Trend detection, demand forecasting
- **Updates**: Dynamic market insights and recommendations

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

## 👥 Team: The Algorithmic Alchemists

| Name | Role | Email |
|------|------|-------|
| **Omprakash Panda** | Team Leader & Full-Stack Developer | omprakash11273@gmail.com |
| **Vishwajith Chakravarthy** | ML Engineer & Data Scientist | krishivishwajith@gmail.com |
| **Vittal Bhajantri** | Backend Developer & DevOps | vittalgb2005@gmail.com |
| **Sindhu B L** | Frontend Developer & UI/UX | sindhublava3105@gmail.com |

## 🏆 Hackathon Details

- **Event**: GenAI Exchange Hackathon
- **Problem Statement**: Personalized Career and Skills Advisor
- **Objective**: Leverage AI to design an innovative solution for personalized career guidance for Indian students
- **Focus**: Dynamic, personalized, and insightful advisory tool that adapts to the fast-changing professional landscape

## 🎯 Key Innovations

1. **Multi-Modal Assessment**: Combines personality, skills, and preferences analysis
2. **Real-Time Market Intelligence**: Dynamic job market analysis with salary predictions
3. **Personalized Learning Paths**: AI-generated roadmaps with specific skill recommendations
4. **Offline-First Architecture**: No dependency on external APIs for core functionality
5. **Indian Job Market Focus**: Tailored for Indian students and job market dynamics

## 📊 Technical Achievements

- **525+ Career Profiles** processed and analyzed
- **3 ML Models** trained and deployed (Career, Salary, Skills)
- **40+ Skill Categories** with intelligent recommendations
- **Big Five Personality Model** implementation
- **Real-time Predictions** with sub-second response times

## 🔮 Future Roadmap

### Phase 1 (Current)
- ✅ Core ML models and web platform
- ✅ Personality and skills assessment
- ✅ Career matching and salary prediction

### Phase 2 (Next 3 months)
- 🔄 Google Cloud integration for scalability
- 🔄 Advanced NLP for resume analysis
- 🔄 Mobile application development
- 🔄 Enterprise partnerships

### Phase 3 (6+ months)
- 📋 AI-powered interview preparation
- 📋 Corporate talent matching platform
- 📋 Advanced analytics dashboard
- 📋 Multi-language support

## 📞 Contact & Support

- **Team Email**: omprakash11273@gmail.com
- **Project Repository**: [GitHub Link]
- **Demo Video**: [Coming Soon]
- **Live Demo**: [Deployment Link]

## 🙏 Acknowledgments

- **GenAI Exchange Hackathon** organizers for the opportunity
- **Google Cloud** for the platform and tools
- **Open Source Community** for ML libraries and frameworks
- **Indian Education System** insights and feedback

---

**Built with ❤️ by The Algorithmic Alchemists**  
*Empowering Indian students with AI-driven career guidance*

**GenAI Exchange Hackathon 2024 Submission**