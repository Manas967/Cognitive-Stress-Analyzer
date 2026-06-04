# 🧠 Cognitive Stress Analyzer

> An AI-powered web application that combines Machine Learning with Google Gemini's Explainable AI to assess and diagnose cognitive stress levels from physiological and contextual inputs.

**MCA-3 Project 

🚀 **Live Demo:** [Click here to use the web application](https://cognitive-stress-analyzer.streamlit.app/)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
  - [Method 1: Virtual Environment (venv)](#method-1-virtual-environment-venv)
  - [Method 2: Docker](#method-2-docker)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Known Issues & Fixes](#known-issues--fixes)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Cognitive Stress Analyzer** is a Streamlit-based web application that predicts a user's stress level using a trained **Random Forest Classifier** and then generates a detailed, clinically-informed diagnostic report via the **Google Gemini 2.5 Flash** language model.

It bridges objective physiological data (heart rate, sleep) with subjective environmental context (noise, work duration, and user-reported feelings) to deliver a holistic stress assessment.

---

## Features

- 🫀 **Heart Rate Estimation** — Calculates BPM from a simple 15-second pulse count
- 🌙 **Sleep & Work Duration Tracking** — Captures lifestyle factors contributing to stress
- 🔊 **Environmental Noise Input** — Accounts for ambient stressors
- 🤖 **ML-based Stress Classification** — Three-tier prediction: Low / Moderate / High Stress
- 🧬 **Gemini XAI Report** — Explainable AI diagnostics that contextualize and validate the ML output
- 📦 **Auto Model Training** — Trains and saves the ML pipeline on first run if models are absent

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| ML Model | Scikit-learn (Random Forest Classifier) |
| Feature Scaling | Scikit-learn (StandardScaler) |
| Generative AI | Google Gemini 2.5 Flash (`google-genai`) |
| Data Processing | Pandas, NumPy |
| Environment Config | python-dotenv |
| Containerization | Docker |

---

## Project Structure

```
cognitive-stress-analyzer/
│
├── app.py                   # Main Streamlit application entry point
├── stress_data.csv          # Training dataset
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker containerization config
├── .env                     # Environment variables (not committed — see Configuration)
│
├── services/
│   ├── gemini_engine.py     # Google Gemini API integration & report generation
│   └── model.py             # ML pipeline: training, scaling, and model persistence
│
└── models/                  # Auto-generated on first run
    ├── stress_model.pkl     # Trained Random Forest model
    └── scaler.pkl           # Fitted StandardScaler
```

> **Note:** The `models/` directory is created automatically when the app is launched for the first time.

---

## Dataset

The training data (`stress_data.csv`) contains 10 labeled samples with the following features:

| Feature | Description |
|---|---|
| `heart_rate` | Heart rate in BPM |
| `work_duration_hours` | Continuous work hours |
| `environment_noise` | Noise level: 1 (Low), 2 (Medium), 3 (High) |
| `sleep_hours` | Hours of sleep the previous night |
| `stress_level` | Label: 0 (Neutral/Low), 1 (Moderate), 2 (High) |

> The dataset is intentionally compact for demonstration. For production use, it is recommended to expand the dataset significantly.

---

## Installation

### Prerequisites

- Python **3.10+**
- A **Google Gemini API Key** — get one free at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Docker (only for Method 2)

---

### Method 1: Virtual Environment (venv)

This is the recommended approach for local development.

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/cognitive-stress-analyzer.git
cd cognitive-stress-analyzer
```

**2. Create and activate a virtual environment**

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:

```bash
# .env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

**5. Ensure the services folder structure is correct**

The app imports from `services/`. Make sure `gemini_engine.py` and `model.py` are inside a `services/` subdirectory with an `__init__.py` file:

```bash
mkdir -p services
touch services/__init__.py
mv gemini_engine.py services/
mv model.py services/
```

**6. Run the application**

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

### Method 2: Docker

Use this method for a fully isolated, reproducible environment.

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/cognitive-stress-analyzer.git
cd cognitive-stress-analyzer
```

**2. Set up environment variables**

Create a `.env` file in the project root:

```bash
# .env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

**3. Build the Docker image**

```bash
docker build -t cognitive-stress-analyzer .
```

**4. Run the container**

```bash
docker run -p 8501:8501 --env-file .env cognitive-stress-analyzer
```

The app will be available at `http://localhost:8501`

> **Tip:** To run in detached (background) mode, add the `-d` flag:
> ```bash
> docker run -d -p 8501:8501 --env-file .env cognitive-stress-analyzer
> ```

---

## Configuration

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) |

Store this in a `.env` file at the project root. **Never commit your `.env` file to version control.**

Add `.env` to your `.gitignore`:

```bash
echo ".env" >> .gitignore
```

---

## Usage

1. **Enter your 15-second pulse count** — the app calculates your BPM automatically.
2. **Set your sleep hours** using the slider.
3. **Provide work and environment context** — how long you've been working and your noise level.
4. **Describe how you're feeling** in the free-text box.
5. **Click "Analyze Stress Level"** — the ML model classifies your stress, and Gemini generates a personalized diagnostic report.

---

## How It Works

```
User Input
    │
    ▼
┌─────────────────────────────┐
│  Feature Engineering        │  pulse_count × 4 → BPM
│  Input Normalization        │  StandardScaler (pre-fitted)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Random Forest Classifier   │  Trained on stress_data.csv
│  Prediction                 │  → 0: Low | 1: Moderate | 2: High
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Gemini 2.5 Flash           │  XAI Diagnostic Report
│  Explainable AI Report      │  Validates & contextualizes ML output
└─────────────────────────────┘
```

The ML model is trained on first launch and cached as `.pkl` files under `models/`. Subsequent runs load the pre-trained artifacts directly.

---

## Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is submitted as an academic project for MCA-3. All rights belong to the respective authors.

---

<p align="center"> Made with ❤️ by Manas Tiwari </p>
