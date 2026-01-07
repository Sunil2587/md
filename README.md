# MindCare - Mental Health Analysis

A production-ready AI-powered mental health analysis application.

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run mp_rds/dashboard/app.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Access:
# - Dashboard: http://localhost:8501
# - API: http://localhost:8000
```

## ☁️ Cloud Deployment

### Option 1: Streamlit Cloud (Recommended - Free)
1. Push this code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `mp_rds/dashboard/app.py`
5. Deploy!

### Option 2: Railway
1. Push to GitHub
2. Go to [railway.app](https://railway.app)
3. Create new project from GitHub
4. Deploy automatically

### Option 3: Heroku
```bash
heroku create your-app-name
git push heroku main
```

### Option 4: Docker (Any Cloud)
```bash
docker build -t mindcare .
docker run -p 8501:8501 mindcare streamlit run mp_rds/dashboard/app.py
```

## 📁 Project Structure
```
├── mp_rds/
│   ├── dashboard/app.py    # Main Streamlit app
│   ├── api/server.py       # FastAPI backend
│   ├── features/           # ML feature extraction
│   └── models/             # Transformer model
├── Dockerfile
├── docker-compose.yml
├── Procfile               # For Heroku/Railway
├── requirements.txt
└── .streamlit/config.toml
```

## ⚠️ Disclaimer
This tool is for self-reflection only and is NOT a medical diagnosis.
Please consult a mental health professional for proper evaluation.
