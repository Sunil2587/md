"""
MindCare - Professional Mental Health Analysis Dashboard
Premium UI with SVG icons instead of emojis
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict

st.set_page_config(
    page_title="MindCare",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# SVG Icons
ICONS = {
    'heart': '''<svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="url(#heartGrad)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><defs><linearGradient id="heartGrad" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#10b981"/><stop offset="100%" style="stop-color:#34d399"/></linearGradient></defs><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>''',
    'edit': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>''',
    'list': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line><line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line></svg>''',
    'activity': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>''',
    'chart': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>''',
    'trending': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>''',
    'lightbulb': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="9" y1="18" x2="15" y2="18"></line><line x1="10" y1="22" x2="14" y2="22"></line><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"></path></svg>''',
    'zap': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>''',
    'cloud': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z"></path></svg>''',
    'broken_heart': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"></path><line x1="12" y1="8" x2="12" y2="16"></line></svg>''',
    'battery_low': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="1" y="6" width="18" height="12" rx="2" ry="2"></rect><line x1="23" y1="13" x2="23" y2="11"></line><line x1="5" y1="10" x2="5" y2="14"></line></svg>''',
    'user_x': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="8.5" cy="7" r="4"></circle><line x1="18" y1="8" x2="23" y2="13"></line><line x1="23" y1="8" x2="18" y2="13"></line></svg>''',
    'wind': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"></path></svg>''',
    'moon': '''<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>''',
    'smile': '''<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M8 14s1.5 2 4 2 4-2 4-2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>''',
    'meh': '''<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="8" y1="15" x2="16" y2="15"></line><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>''',
    'frown': '''<svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M16 16s-1.5-2-4-2-4 2-4 2"></path><line x1="9" y1="9" x2="9.01" y2="9"></line><line x1="15" y1="9" x2="15.01" y2="9"></line></svg>''',
    'phone': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>''',
    'alert': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>''',
    'message': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>''',
    'walk': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="5" r="2"></circle><path d="M10 22v-5l-2-3v-6h8v6l-2 3v5"></path></svg>''',
    'breath': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"></path><path d="M12 6v6l4 2"></path></svg>''',
    'book': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>''',
    'bed': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7v11m0-4h18m0 4V8a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v3"></path><path d="M7 14h.01M7 18h.01"></path></svg>''',
    'music': '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>'''
}

# Premium CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

#MainMenu, footer, header, .stDeployButton, div[data-testid="stToolbar"], div[data-testid="stDecoration"] {display: none !important;}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
}

/* Hero */
.hero {
    text-align: center;
    padding: 50px 20px 40px;
    position: relative;
}

.hero-icon {
    margin-bottom: 20px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}

.hero-subtitle {
    font-size: 18px;
    color: #64748b;
    font-weight: 400;
}

/* Cards */
.card {
    background: rgba(30, 41, 59, 0.5);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 24px;
    padding: 30px;
    margin: 15px 0;
}

.card:hover {
    border-color: rgba(139, 92, 246, 0.3);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 15px;
    margin-bottom: 20px;
}

.card-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
}

/* Input */
.stTextArea textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-size: 15px !important;
    padding: 18px !important;
}

.stTextArea textarea:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 16px 40px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 40px rgba(139, 92, 246, 0.4) !important;
}

/* Status */
.status-card {
    border-radius: 24px;
    padding: 45px;
    text-align: center;
    margin: 25px 0;
}

.status-healthy {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.08));
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.status-warning {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.08));
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.status-alert {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.15), rgba(239, 68, 68, 0.08));
    border: 1px solid rgba(244, 63, 94, 0.3);
}

.status-icon { margin-bottom: 20px; }
.status-title { font-size: 32px; font-weight: 700; color: #f1f5f9; margin-bottom: 10px; }
.status-desc { font-size: 16px; color: #94a3b8; }

/* Insight */
.insight-card {
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
}

.insight-header {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 15px;
}

.insight-icon { color: #a78bfa; flex-shrink: 0; }
.insight-title { font-size: 17px; font-weight: 600; color: #f1f5f9; margin-bottom: 4px; }
.insight-desc { font-size: 13px; color: #94a3b8; line-height: 1.5; }

/* Guide section */
.guide-section {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.15);
    border-radius: 14px;
    padding: 18px 20px;
    margin-top: 15px;
}

.guide-title {
    font-size: 13px;
    font-weight: 600;
    color: #10b981;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.guide-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.guide-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.5;
}

.guide-bullet {
    color: #10b981;
    font-weight: bold;
    flex-shrink: 0;
}


/* Suggestions */
.suggestion-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: #e2e8f0;
    padding: 12px 20px;
    border-radius: 50px;
    font-size: 14px;
    margin: 6px;
}

/* Metrics */
.metric-box {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 25px;
    text-align: center;
    margin-bottom: 15px;
}

.metric-value {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label { font-size: 12px; color: #64748b; margin-top: 5px; text-transform: uppercase; letter-spacing: 1px; }

/* Message */
.message-item {
    background: rgba(15, 23, 42, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 12px;
    padding: 15px;
    margin: 10px 0;
}

.message-num { font-size: 11px; color: #8b5cf6; font-weight: 600; margin-bottom: 6px; }
.message-text { color: #cbd5e1; font-size: 13px; line-height: 1.5; }

/* Empty */
.empty-state { text-align: center; padding: 60px; color: #475569; }
.empty-icon { margin-bottom: 15px; opacity: 0.5; }

/* Crisis */
.crisis-card {
    background: rgba(244, 63, 94, 0.08);
    border: 1px solid rgba(244, 63, 94, 0.25);
    border-radius: 20px;
    padding: 30px;
    margin-top: 25px;
}

.crisis-header { display: flex; align-items: center; gap: 12px; margin-bottom: 15px; }
.crisis-title { color: #f43f5e; font-size: 18px; font-weight: 700; }
.crisis-text { color: #94a3b8; margin-bottom: 18px; font-size: 14px; }
.crisis-pill { background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.25); color: #fda4af; padding: 10px 18px; border-radius: 50px; font-size: 13px; margin: 5px; display: inline-flex; align-items: center; gap: 8px; }

/* Footer */
.footer { text-align: center; padding: 40px; color: #475569; font-size: 12px; }
</style>
""", unsafe_allow_html=True)


def analyze(messages: List[str]) -> Dict:
    text = " ".join(messages).lower()
    
    patterns = {
        'hopelessness': ['hopeless', 'no hope', 'pointless', 'meaningless', 'give up', 'no point'],
        'sadness': ['sad', 'crying', 'miserable', 'unhappy', 'depressed', 'hurt', 'pain'],
        'fatigue': ['tired', 'exhausted', 'no energy', 'drained', 'weak'],
        'isolation': ['alone', 'lonely', 'nobody', 'isolated', 'no friends', 'no one'],
        'anxiety': ['anxious', 'worried', 'panic', 'nervous', 'scared', 'overwhelmed'],
        'sleep': ['can\'t sleep', 'insomnia', 'nightmares', 'awake', 'sleepless']
    }
    
    positive = ['happy', 'good', 'great', 'hope', 'better', 'love', 'grateful']
    
    scores = {}
    risk = 0
    for cat, words in patterns.items():
        count = sum(1 for w in words if w in text)
        scores[cat] = min(count * 25, 100)
        risk += count
    
    pos = sum(1 for w in positive if w in text)
    wellness = max(0, min(100, 100 - (risk * 12) + (pos * 10)))
    
    words_list = text.split()
    self_focus = (sum(1 for w in words_list if w in ['i', 'me', 'my', 'myself']) / max(len(words_list), 1)) * 100
    
    if wellness >= 60:
        status, icon, title, desc = 'healthy', 'smile', "You're Doing Great!", "Your messages indicate a healthy mental state. Keep taking care of yourself!"
    elif wellness >= 35:
        status, icon, title, desc = 'warning', 'meh', "We're Here For You", "We noticed some signs you might be going through a tough time."
    else:
        status, icon, title, desc = 'alert', 'frown', "You Deserve Support", "Your words show you may be struggling. Please know you're not alone."
    
    insights = []
    insight_data = {
        'hopelessness': {
            'icon': 'cloud',
            'title': 'Feeling Hopeless',
            'desc': 'You expressed feelings of hopelessness or lack of purpose.',
            'guide': [
                'Remember that feelings are temporary, not permanent facts',
                'Set one tiny goal for today - just one small win',
                'Write down 3 things that went okay today, no matter how small',
                'Reach out to one person who cares about you',
                'Consider speaking with a therapist - it really helps'
            ]
        },
        'sadness': {
            'icon': 'broken_heart',
            'title': 'Experiencing Sadness',
            'desc': 'There are signs of emotional pain in what you shared.',
            'guide': [
                'Allow yourself to feel - crying is healthy and healing',
                'Listen to music that matches or gently lifts your mood',
                'Spend 10 minutes in sunlight or nature today',
                'Talk to a friend, family member, or counselor',
                'Practice self-compassion - treat yourself like you would a friend'
            ]
        },
        'fatigue': {
            'icon': 'battery_low',
            'title': 'Low Energy',
            'desc': 'You mentioned feeling exhausted or lacking motivation.',
            'guide': [
                'Start with just 5 minutes of light movement or stretching',
                'Check your sleep schedule - aim for 7-9 hours',
                'Stay hydrated and eat regular, nutritious meals',
                'Break tasks into tiny steps - do just the first one',
                'Rule out medical causes with a doctor visit'
            ]
        },
        'isolation': {
            'icon': 'user_x',
            'title': 'Feeling Isolated',
            'desc': 'You seem to be experiencing loneliness or disconnection.',
            'guide': [
                'Send a simple "thinking of you" text to someone today',
                'Join an online community around your interests',
                'Visit a public place like a cafe or library',
                'Consider volunteering - helping others builds connection',
                'Schedule regular check-ins with friends or family'
            ]
        },
        'anxiety': {
            'icon': 'wind',
            'title': 'Anxiety Signs',
            'desc': 'There are indicators of worry or overwhelming stress.',
            'guide': [
                'Try the 5-4-3-2-1 grounding technique (5 things you see, 4 hear, etc.)',
                'Practice box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s',
                'Limit caffeine and reduce screen time before bed',
                'Write down your worries to get them out of your head',
                'Challenge anxious thoughts - ask "Is this really likely?"'
            ]
        },
        'sleep': {
            'icon': 'moon',
            'title': 'Sleep Issues',
            'desc': 'You mentioned having trouble with sleep.',
            'guide': [
                'Set a consistent bedtime and wake time every day',
                'Avoid screens 1 hour before bed - try reading instead',
                'Keep your bedroom cool, dark, and quiet',
                'Try relaxation techniques like progressive muscle relaxation',
                'Limit naps to 20 minutes and avoid them after 3 PM'
            ]
        }
    }
    
    for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if score >= 20 and cat in insight_data:
            data = insight_data[cat]
            insights.append({
                'icon': data['icon'],
                'title': data['title'],
                'desc': data['desc'],
                'guide': data['guide']
            })

    
    suggestions = [
        ('message', 'Talk to someone you trust'),
        ('walk', 'Take a mindful walk'),
        ('breath', 'Try 5 min meditation'),
        ('book', 'Journal your thoughts'),
        ('bed', 'Prioritize sleep'),
        ('music', 'Listen to calming music')
    ] if wellness < 60 else []
    
    return {
        'wellness': wellness,
        'status': status,
        'icon': icon,
        'title': title,
        'desc': desc,
        'scores': scores,
        'insights': insights[:4],
        'suggestions': suggestions,
        'count': len(messages),
        'self_focus': self_focus
    }


def gauge_chart(score):
    color = "#10b981" if score >= 60 else "#fbbf24" if score >= 35 else "#f43f5e"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '%', 'font': {'size': 52, 'color': 'white', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'rgba(0,0,0,0)'},
            'bar': {'color': color, 'thickness': 0.22},
            'bgcolor': 'rgba(255,255,255,0.03)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': 'rgba(244, 63, 94, 0.06)'},
                {'range': [35, 60], 'color': 'rgba(251, 191, 36, 0.06)'},
                {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.06)'}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=260, margin=dict(l=25, r=25, t=40, b=25))
    return fig


def radar_chart(scores):
    cats = [c.title() for c in scores.keys()]
    vals = list(scores.values())
    cats.append(cats[0])
    vals.append(vals[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats, fill='toself',
        fillcolor='rgba(139, 92, 246, 0.2)',
        line=dict(color='#8b5cf6', width=2),
        marker=dict(size=6, color='#8b5cf6')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showline=False, tickfont=dict(color='rgba(255,255,255,0.25)', size=9)),
            angularaxis=dict(tickfont=dict(color='rgba(255,255,255,0.5)', size=11)),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=70, r=70, t=35, b=35)
    )
    return fig


# ========== APP ==========

st.markdown(f'''
<div class="hero">
    <div class="hero-icon">{ICONS['heart']}</div>
    <div class="hero-title">MindCare</div>
    <div class="hero-subtitle">AI-Powered Mental Wellness Analysis</div>
</div>
''', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(f'''
    <div class="card">
        <div class="card-header">
            <div class="card-icon">{ICONS['edit']}</div>
            <div class="card-title">Share Your Thoughts</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Messages",
        height=220,
        placeholder="Write about your feelings or paste journal entries here...\n\nExample:\nDay 1: Started the week okay.\nDay 3: Work stress is getting to me.",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze My Wellbeing", use_container_width=True)

with col2:
    st.markdown(f'''
    <div class="card">
        <div class="card-header">
            <div class="card-icon">{ICONS['list']}</div>
            <div class="card-title">Message Preview</div>
        </div>
    ''', unsafe_allow_html=True)
    
    messages = [m.strip() for m in user_input.split('\n') if m.strip()] if user_input else []
    
    if messages:
        for i, msg in enumerate(messages[:4]):
            st.markdown(f'<div class="message-item"><div class="message-num">Entry {i+1}</div><div class="message-text">{msg[:100]}{"..." if len(msg) > 100 else ""}</div></div>', unsafe_allow_html=True)
        if len(messages) > 4:
            st.markdown(f'<p style="color:#64748b;text-align:center;margin-top:12px;">+ {len(messages)-4} more</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="empty-state"><div class="empty-icon">{ICONS["list"]}</div><div>Your thoughts will appear here</div></div>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

if analyze_btn and messages:
    result = analyze(messages)
    
    st.markdown(f'''
    <div class="status-card status-{result['status']}">
        <div class="status-icon">{ICONS[result['icon']]}</div>
        <div class="status-title">{result['title']}</div>
        <div class="status-desc">{result['desc']}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")
    
    with c1:
        st.markdown(f'<div class="card"><div class="card-header"><div class="card-icon">{ICONS["activity"]}</div><div class="card-title">Wellness Score</div></div>', unsafe_allow_html=True)
        st.plotly_chart(gauge_chart(result['wellness']), use_container_width=True, key="g")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown(f'<div class="card"><div class="card-header"><div class="card-icon">{ICONS["chart"]}</div><div class="card-title">Pattern Analysis</div></div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(result['scores']), use_container_width=True, key="r")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c3:
        st.markdown(f'''
        <div class="card">
            <div class="card-header">
                <div class="card-icon">{ICONS['trending']}</div>
                <div class="card-title">Quick Stats</div>
            </div>
            <div class="metric-box"><div class="metric-value">{result['count']}</div><div class="metric-label">Messages</div></div>
            <div class="metric-box"><div class="metric-value">{result['self_focus']:.0f}%</div><div class="metric-label">Self-Focus</div></div>
        </div>
        ''', unsafe_allow_html=True)
    
    if result['insights']:
        st.markdown(f'<div class="card"><div class="card-header"><div class="card-icon">{ICONS["lightbulb"]}</div><div class="card-title">What We Found & How To Improve</div></div>', unsafe_allow_html=True)
        for ins in result['insights']:
            guide_items = "".join([f'<div class="guide-item"><span class="guide-bullet">→</span><span>{tip}</span></div>' for tip in ins.get('guide', [])])
            guide_html = f'<div class="guide-section"><div class="guide-title">Steps to Feel Better</div><div class="guide-list">{guide_items}</div></div>' if guide_items else ''
            st.markdown(f'''
            <div class="insight-card">
                <div class="insight-header">
                    <div class="insight-icon">{ICONS[ins["icon"]]}</div>
                    <div>
                        <div class="insight-title">{ins["title"]}</div>
                        <div class="insight-desc">{ins["desc"]}</div>
                    </div>
                </div>
                {guide_html}
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    if result['suggestions']:
        pills = "".join([f'<span class="suggestion-pill">{ICONS[s[0]]}<span>{s[1]}</span></span>' for s in result['suggestions']])
        st.markdown(f'<div class="card"><div class="card-header"><div class="card-icon">{ICONS["zap"]}</div><div class="card-title">Things That Might Help</div></div>{pills}</div>', unsafe_allow_html=True)
    
    if result['status'] == 'alert':
        st.markdown(f'''
        <div class="crisis-card">
            <div class="crisis-header">{ICONS['alert']}<span class="crisis-title">Need Support Right Now?</span></div>
            <div class="crisis-text">If you're having thoughts of hurting yourself, please reach out. You matter.</div>
            <div>
                <span class="crisis-pill">{ICONS['phone']} iCall: 9152987821</span>
                <span class="crisis-pill">{ICONS['phone']} 988 Lifeline</span>
                <span class="crisis-pill">{ICONS['phone']} findahelpline.com</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<div class="footer">MindCare Analytics • For self-reflection only • Not a medical diagnosis</div>', unsafe_allow_html=True)
