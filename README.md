# MindCare - Mental Health Analysis Dashboard

AI-powered mental health self-reflection tool that analyzes your journal entries and provides wellness insights with actionable recovery guides.

## Features

- **Wellness Score** - Overall mental health score (0-100%)
- **Pattern Analysis** - Radar chart showing emotional patterns
- **Smart Detection** - Identifies hopelessness, sadness, anxiety, isolation, fatigue, sleep issues
- **Recovery Guides** - 5 actionable tips for each detected issue
- **Beautiful UI** - Modern dark theme with SVG icons

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Sunil2587/md.git
cd md
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard

```bash
streamlit run mp_rds/dashboard/app.py
```

The dashboard will open at: **http://localhost:8501**

## How to Use

1. Open the dashboard in your browser
2. Paste your journal entries or thoughts in the text area (one per line)
3. Click **"Analyze My Wellbeing"**
4. View your wellness score, pattern analysis, and personalized recovery tips

## Project Structure

```
md/
├── mp_rds/
│   ├── dashboard/
│   │   └── app.py          # Main Streamlit dashboard
│   ├── api/
│   │   └── server.py       # FastAPI REST API
│   ├── features/           # ML feature extraction
│   ├── models/             # Transformer model & RDS engine
│   └── config/             # Configuration settings
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- Streamlit
- Plotly
- NumPy

## API Usage (Optional)

Run the API server:
```bash
uvicorn mp_rds.api.server:app --host 0.0.0.0 --port 8000
```

API Docs: **http://localhost:8000/docs**

## Disclaimer

This tool is for **self-reflection only** and is NOT a medical diagnosis. If you're struggling with mental health, please consult a professional.

**Crisis Helplines:**
- India: iCall 9152987821
- USA: 988 Lifeline
- International: findahelpline.com

## License

MIT License
