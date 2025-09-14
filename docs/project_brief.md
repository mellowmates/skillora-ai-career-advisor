
# Project Brief: Skillora AI Career Advisor

## 1. Overview

Skillora is an **AI-powered career guidance platform** designed to provide personalized career and skills advice to students in India. It addresses the challenge of generic career counseling by offering data-driven, individualized recommendations based on user assessments and real-time job market trends. The platform uses Google's Gemini AI to power its core recommendation engine and a conversational chatbot, ensuring that the guidance is both intelligent and relevant to the current professional landscape.

> This project was developed as a submission for the **GenAI Exchange Hackathon** by the team _"The Algorithmic Alchemists."_

---

## 2. Core Problem & Solution

**Problem:**
> Students in India often struggle with career choices due to a lack of personalized guidance that accounts for their unique skills, interests, and the dynamic nature of the job market.

**Solution:**
Skillora provides a personalized, interactive platform that:

- **Assesses Users:** Gathers information on a user's experience, education, interests, and skills.
- **Generates Recommendations:** Uses the Gemini AI model to analyze the user's profile against current market data to suggest top career matches.
- **Offers Skill Development Plans:** Provides actionable advice on which skills to develop.
- **Provides AI Chat:** Includes an AI-powered chatbot for users to ask specific career-related questions.

---

## 3. Key Features

- **Interactive Multi-Step Assessment:** A user-friendly questionnaire to collect essential user data.

- **AI-Powered Analysis:**
  - The core of the application is an API endpoint (`/api/analyze`) that sends a detailed prompt with user assessment data and market trends to the Gemini API. The AI is instructed to return a structured JSON object containing:
	 - `top_matches`: A list of the top 3 recommended careers with match percentage, salary, and growth potential.
	 - `skill_development`: A list of recommended skills with proficiency goals.
	 - `market_insights`: Key metrics about the job market tailored to the user's profile.

- **Personalized Dashboard:** A dynamic dashboard that visualizes the AI-generated recommendations, skill plans, and market insights.

- **Conversational AI Chatbot:** An AI-powered chat interface (`/api/chat`) that allows users to have a natural conversation with "Skillora" to get answers to their career questions.

- **Dark Mode Toggle:** A user-friendly option to switch between light and dark themes.

---

## 4. Technical Architecture & Stack

Skillora is a single-page web application with a Python backend.

### Backend

- **Framework:** Flask
- **Language:** Python
- **AI Integration:** `google-generativeai` SDK for Python to interact with the `gemini-1.5-flash` model.
- **Environment Management:** `python-dotenv` to manage the Gemini API key securely.

### Frontend

- **Structure:** A single `index.html` file.
- **Styling:** Tailwind CSS for a modern, responsive UI.
- **Interactivity:** Alpine.js for handling UI state, such as tab switching, form data, and API calls.
- **Icons:** Lucide Icons for a clean and modern look.

### Data

- **Job Market Data:** `data/job_market_data.json` provides high-level market statistics.
- **Career Profiles:** `data/career_profiles.json` contains foundational information about different career paths, though the primary recommendation logic is handled by the AI.

---

## 5. How It Works

1. **User Assessment:**
	- The user completes a multi-step form on the frontend, providing details about their experience, education, interests, skills, and goals. This data is managed by Alpine.js.

2. **API Request:**
	- Upon completing the assessment, the frontend sends a POST request to the `/api/analyze` endpoint with the user's data in JSON format.

3. **Prompt Engineering:**
	- The Flask backend receives the data, combines it with the pre-loaded market data, and constructs a detailed prompt. This prompt instructs the Gemini model to act as "Skillora" and return a structured JSON response with career matches, skill plans, and market insights.

4. **AI Response & Dashboard:**
	- The backend sends the AI's JSON response back to the frontend. Alpine.js then uses this data to dynamically populate the user's personalized dashboard.

5. **AI Chat:**
	- The user can interact with the chatbot. Each message is sent to the `/api/chat` endpoint, where another prompt is engineered to make the Gemini model act as a helpful career advisor, and the response is displayed in the chat interface.