# Skillora AI Career Advisor

**GenAI Exchange Hackathon Submission - Team: The Algorithmic Alchemists**

A comprehensive AI-powered career guidance platform that leverages machine learning to analyze personality traits, skills, and market trends to provide personalized career recommendations for Indian students navigating the evolving job market.

## 🎯 Problem Statement

**Personalized Career and Skills Advisor**

Students in India often face a bewildering array of career choices, compounded by generic guidance that fails to account for their unique interests, aptitudes, and the rapidly evolving job market. The traditional approach to career counseling struggles to keep pace with the emergence of new job roles and the specific skills required to succeed in them.

## 🚀 Our Solution

Skillora AI addresses this challenge by providing a smart, personalized, and data-driven career guidance platform. Our solution empowers users by:

- Analyzing their unique skills, interests, and educational background through an interactive assessment.
- Recommending the most suitable career paths using a rule-based engine that considers user data and real-time market trends.
- Providing direct access to an AI-powered career coach (via the Gemini API) for intelligent, conversational guidance on any career-related topic.

This approach ensures that every user receives advice that is not only personalized but also relevant to the current demands of the Indian job market.

## ⚠️ Prototype Disclaimer

Please note that this project is a functional prototype developed within the time constraints of the GenAI Exchange Hackathon. It is designed to demonstrate the core features and the technical viability of the Skillora AI concept. It is not a complete, production-ready application. The focus has been on the user interface, the AI chatbot integration, and the career recommendation logic.

## ✨ Core Features

This prototype demonstrates the key features that make Skillora a powerful career guidance tool:

- **Interactive User Assessment:** A multi-step questionnaire that dynamically gathers information about a user's education, interests, and skills.
- **Intelligent Career Advisor:** A backend engine that analyzes assessment data to provide a ranked list of the top 3 most suitable career paths, complete with a match score.
- **Real-Time Market Insights:** The dashboard displays key data points about the current job market to help users make informed decisions.
- **AI Chatbot (Powered by Gemini):** A conversational AI coach that connects to the Google Cloud Gemini API to provide intelligent, context-aware answers to career-related questions.

## ⚙️ Technology Stack

This prototype is built with a modern and efficient technology stack, with a strong focus on meeting the hackathon's technical evaluation criteria.

### Backend:

- **Python:** The core programming language.
- **Flask:** A lightweight web framework to serve the application and create API endpoints.
- **Google Generative AI:** The official Python SDK to connect to the Gemini API.

### Frontend:

- **HTML, Tailwind CSS, Alpine.js:** A single-page application experience built with modern web technologies for a responsive and beautiful UI.

### AI Integration:

- **Google Cloud Gemini API:** The `gemini-2.5-flash` model is used to power the AI Chatbot, fulfilling the "AI tool utilization" requirement.

## 🛠️ Setup and Installation

Follow these steps to get the Skillora prototype running on your local machine.

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A Google Gemini API Key

### 1. Clone the Repository

```bash
git clone https://github.com/mellowmates/skillora-ai-career-advisor.git
cd skillora-ai-career-advisor
```

### 2. Set up a Virtual Environment

It's a best practice to use a virtual environment to manage project dependencies.

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all the necessary Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

Create a new file named `.env` in the root of your project folder. This file will securely store your API key.

Open the `.env` file and add the following line, replacing `your_actual_api_key_here` with your real Gemini API key:

```
GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Run the Application

You are now ready to start the Flask application.

```bash
python app.py
```

The application will be running at `http://127.0.0.1:5000`. Open this address in your web browser to see the prototype in action.

## 🤖 AI Integration with Google Gemini

The core AI feature of this prototype is the chatbot, which is powered by the Google Gemini API. This integration is handled in the `app.py` backend.

- **Secure API Key:** The application securely loads the `GEMINI_API_KEY` from the `.env` file, ensuring no sensitive credentials are hardcoded in the source code.
- **API Endpoint:** A Flask route at `/api/chat` is dedicated to handling chatbot requests from the frontend.
- **Prompt Engineering:** When a user sends a message, the backend constructs a carefully crafted prompt. This prompt instructs the `gemini-2.5-flash` model to act as "Skillora," a friendly and professional career advisor for the Indian market.
- **Live AI Response:** The prompt is sent to the Gemini API, which generates an intelligent, context-aware response. This response is then sent back to the user in the chat interface.

This approach demonstrates a powerful and effective use of a state-of-the-art Large Language Model to provide a dynamic and helpful user experience.

## 🏆 Hackathon Submission

- **Event:** GenAI Exchange Hackathon
- **Problem Statement:** Personalized Career and Skills Advisor
- **Objective:** To leverage AI to design an innovative solution for personalized career guidance for Indian students, creating a dynamic and insightful advisory tool that adapts to the fast-changing professional landscape.

## 👥 Team: The Algorithmic Alchemists

| Name                   | Role                             |
| ---------------------- | -------------------------------- |
| Omprakash Panda        | Team Leader & AI/ML Development  |
| Vishwajith Chakravarthy| Testing& Documentation           |
| Vittal Bhajantri       | Backend Integration              |
| Sindhu B L             | Frontend & UI/UX                 |
| Manoj R                | Presentation& Video Editing      |

## 📞 Contact & Support

- **Team Email:** [omprakash11273@gmail.com](mailto:omprakash11273@gmail.com)
- **Project Repository:** [https://github.com/mellowmates/skillora-ai-career-advisor](https://github.com/mellowmates/skillora-ai-career-advisor)

## 🙏 Acknowledgments

- GenAI Exchange Hackathon organizers for the opportunity.
- Google Cloud for the platform and tools that made this project possible.
- The Open Source Community for the amazing libraries and frameworks.

---

*Built with ❤️ by The Algorithmic Alchemists. Empowering Indian students with AI-driven career guidance.*

*GenAI Exchange Hackathon 2025 Submission*
