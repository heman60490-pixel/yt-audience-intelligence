import os
import re
import json
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from fpdf import FPDF
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from groq import Groq

# ---------------------------------------------------------
# 1. Page Configuration & Professional CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube Audience Intelligence Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_styling = """
<style>
    /* Complete cleanup of Streamlit Cloud default headers & badges */
    #MainMenu, header, footer, [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"], div[class*="viewerBadge"], [data-testid="stViewerBadge"] {display: none !important;}
    div[class*="profile"], div[data-testid="stBottomBlockContainer"], [data-testid="stToolbar"] {display: none !important;}
    [class^="FloatingApp"], [class*="viewer_badge"], div[data-testid="stToolbarActions"] {display: none !important;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }
    
    /* SaaS Metric Cards */
    .kpi-card {
        background: linear-gradient(145deg, #161b26, #0f131c);
        border: 1px solid #232d3f;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00d2ff;
    }
    
    /* Insight Cards */
    .insight-card {
        background-color: #161a24;
        border: 1px solid #232a3b;
        border-left: 4px solid #00d2ff;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(custom_styling, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Key Management (Secrets + Sidebar Fallback)
# ---------------------------------------------------------
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
yt_key = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))

if not groq_key or not yt_key:
    with st.sidebar:
        st.subheader("🔑 API Configuration")
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password")
        if not yt_key:
            yt_key = st.text_input("YouTube API Key", type="password")

if not groq_key or not yt_key:
    st.warning("⚠️ Please configure **GROQ_API_KEY** and **YOUTUBE_API_KEY** to start.")
    st.stop()

client = Groq(api_key=groq_key)

# ---------------------------------------------------------
# 3. Data Processing & API Calls
# ---------------------------------------------------------
def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_youtube_data(video_id: str, api_key: str, max_comments: int = 150):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        vid_response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()
        
        if not vid_response.get("items"):
            return None, None, None
        
        item = vid_response["items"][0]
        video_details = item["snippet"]
        video_stats = item["statistics"]
        
        comments = []
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments),
            order="relevance",
            textFormat="plainText"
        )
        while req and len(comments) < max_comments:
            res = req.execute()
            for c_item in res.get("items", []):
                comm = c_item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comm)
            req = youtube.commentThreads().list_next(req, res)
            
        return video_details, video_stats, comments

    except HttpError as e:
        if e.resp.status == 403:
            st.error("⚠️ YouTube API Quota limit reached or invalid API Key.")
        else:
            st.error(f"⚠️ YouTube API Error: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {str(e)}")
        return None, None, None

@st.cache_data(ttl=86400, show_spinner=False)
def run_groq_intelligence(comments: list, video_title: str) -> dict:
    prompt = f"""
    You are an expert Audience Intelligence Analyst.
    Analyze the following {len(comments)} YouTube comments for the video: "{video_title}".
    Note: Understand Hinglish, Hindi in Roman/Devanagari script, slang, and English contextually.

    Provide deep strategic synthesis in pure JSON with keys:
    1. "kpis": {{
         "sentiment_score": (int 0-100),
         "dominant_intent": (string, e.g. "Actionable Learning / Technical Curiosity"),
         "audience_vibe": (string, e.g. "Supportive & Inquisitive")
       }}
    2. "clusters": List of objects with keys "category", "percentage" (number summing up to 100), "summary".
    3. "demand_share": List of specific topics viewers demand most.
    4. "competitor_gaps": Unmet needs, missing explanations, or comparison angles.
    5. "shorts_polls": {{
         "shorts": [
           {{"hook": "string", "core_concept": "string", "cta": "string"}},
           {{"hook": "string", "core_concept": "string", "cta": "string"}},
           {{"hook": "string", "core_concept": "string", "cta": "string"}}
         ],
         "polls": [
           {{"question": "string", "options": ["opt1", "opt2", "opt3", "opt4"]}},
           {{"question": "string", "options": ["opt1", "opt2", "opt3", "opt4"]}}
         ]
       }}
    6. "doubts_myths": Top doubts or misconceptions raised in comments.
    7. "action_blueprint": 4 concrete next steps for the creator.

    Comments:
    {json.dumps(comments[:120], ensure_ascii=False)}

    Output STRICTLY raw JSON only.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def generate_pdf_report(video_title, channel_title, kpis, clusters, blueprint):
    """Generate a clean executive PDF report using FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Audience Intelligence Executive Summary", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, f"Generated for: {video_title[:65]}...", ln=True, align="C")
    pdf.cell(0, 6, f"Channel: {channel_title}", ln=True, align="C")
    pdf.ln(8)
    
    # KPI Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "1. Core Metrics & Audience Vibe", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"- Net Sentiment Score: {kpis.get('sentiment_score', 'N/A')}% Positive", ln=True)
    pdf.cell(0, 6, f"- Dominant Intent: {kpis.get('dominant_intent', 'N/A')}", ln=True)
    pdf.cell(0, 6, f"- Audience Vibe: {kpis.get('audience_vibe', 'N/A')}", ln=True)
    pdf.ln(6)
    
    # Sentiment Clusters
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "2. Intent & Sentiment Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for c in clusters:
        pdf.cell(0, 6, f"- {c.get('category')}: {c.get('percentage')}% | {c.get('summary')}", ln=True)
    pdf.ln(6)
    
    # Action Blueprint
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "3. Next Steps & Growth Blueprint", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for idx, step in enumerate(blueprint, 1):
        pdf.multi_cell(0, 6, f"{idx}. {step}")
        
    return bytes(pdf.output())

# ---------------------------------------------------------
# 4. Header UI
# ---------------------------------------------------------
st.markdown("## ⚡ YouTube Audience Intelligence Suite")
st.caption("AI-Powered Semantic Clustering, Gap Arbitrage & Creator Action Blueprints")

col_url, col_mode, col_btn = st.columns([5, 3, 2])
with col_url:
    yt_input_url = st.text_input("URL", placeholder="Paste YouTube Video URL...", label_visibility="collapsed")
with col_mode:
    mode_selected = st.selectbox("Mode", ["🎯 Single Video Deep Dive", "⚔️ Competitor Gap Mining"], label_visibility="collapsed")
with col_btn:
    analyze_pressed = st.button("🚀 Analyze Insights", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 5. Pipeline Execution
# ---------------------------------------------------------
if analyze_pressed and yt_input_url:
    vid_id = extract_video_id(yt_input_url)
    if not vid_id:
        st.error("Please provide a valid YouTube URL.")
        st.stop()
        
    with st.spinner("Analyzing comments with Groq Llama-3.3-70B..."):
        video_details, video_stats, comments = fetch_youtube_data(vid_id, yt_key)
        
        if not video_details or not comments:
            st.stop()
            
        insights = run_groq_intelligence(comments, video_details.get("title", ""))
        st.session_state["insights"] = insights
        st.session_state["video_details"] = video_details
        st.session_state["video_stats"] = video_stats
        st.session_state["comments"] = comments

if "insights" in st.session_state:
    insights = st.session_state["insights"]
    vid_meta = st.session_state["video_details"]
    vid_stats = st.session_state.get("video_stats", {})
    comments = st.session_state["comments"]
    kpis = insights.get("kpis", {})

    st.markdown("---")
    
    # Video Card
    v1, v2, v3 = st.columns([2, 5, 2])
    with v1:
        st.image(vid_meta["thumbnails"]["high"]["url"], use_container_width=True)
    with v2:
        st.subheader(vid_meta.get("title", ""))
        st.caption(f"Channel: **{vid_meta.get('channelTitle', '')}** | Views: **{int(vid_stats.get('viewCount', 0)):,}**")
    with v3:
        st.markdown(f"#### Analyzed")
        st.markdown(f"### **{len(comments)} Comments**")

    # KPI Metric Cards
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Audience Sentiment Index</div>
            <div class="kpi-value">{kpis.get('sentiment_score', 80)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Dominant Audience Intent</div>
            <div class="kpi-value" style="font-size: 1.1rem; color: #29B5E8; margin-top: 6px;">{kpis.get('dominant_intent', 'Learning')}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Audience Vibe</div>
            <div class="kpi-value" style="font-size: 1.1rem; color: #00CC96; margin-top: 6px;">{kpis.get('audience_vibe', 'Supportive')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Export Bar
    pdf_bytes = generate_pdf_report(
        vid_meta.get("title", ""),
        vid_meta.get("channelTitle", ""),
        kpis,
        insights.get("clusters", []),
        insights.get("action_blueprint", [])
    )
    
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("📥 Download Executive PDF", data=pdf_bytes, file_name="Audience_Report.pdf", mime="application/pdf", use_container_width=True)
    with d2:
        st.download_button("📄 Download JSON Summary", data=json.dumps(insights, indent=2), file_name="insights.json", mime="application/json", use_container_width=True)
    with d3:
        st.download_button("📝 Download Comments (.txt)", data="\n\n".join(comments), file_name="comments.txt", mime="text/plain", use_container_width=True)

    st.markdown("---")

    # Tabs Section
    tabs = st.tabs([
        "📊 Sentiment & Clusters",
        "☁️ Word Cloud & Topics",
        "📈 Demand Share",
        "⚔️ Competitor Gaps",
        "📱 Shorts & Polls Studio",
        "❓ Doubts & Myths",
        "📋 Action Blueprint",
        "💬 Raw Comments"
    ])

    # Tab 1: Sentiment Donut & Cards
    with tabs[0]:
        st.subheader("Audience Intent & Sentiment Clusters")
        clusters = insights.get("clusters", [])
        if clusters:
            df_c = pd.DataFrame(clusters)
            fig = px.pie(
                df_c, names="category", values="percentage", hole=0.55,
                color_discrete_sequence=["#00d2ff", "#FF4B4B", "#FFAA00", "#00CC96"]
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff")
            )
            st.plotly_chart(fig, use_container_width=True)

            for item in clusters:
                st.markdown(f"""
                <div class="insight-card">
                    <h4>{item.get('category')} — {item.get('percentage')}%</h4>
                    <p style="color: #94a3b8; margin: 0;">{item.get('summary')}</p>
                </div>
                """, unsafe_allow_html=True)

    # Tab 2: Word Cloud & Keywords
    with tabs[1]:
        st.subheader("Audience Keyword Cloud & Frequent Terms")
        all_text = " ".join(comments)
        stopwords = set(STOPWORDS)
        stopwords.update(["video", "channel", "sir", "bhai", "hai", "karo", "karein", "aap"])
        
        wc = WordCloud(width=800, height=350, background_color="#0e1117", stopwords=stopwords, colormap="Blues").generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(10, 4.5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig_wc.patch.set_facecolor('#0e1117')
        st.pyplot(fig_wc)

    # Tab 3: Demand Share
    with tabs[2]:
        st.subheader("High-Demand Topic Inquiries")
        for d in insights.get("demand_share", []):
            st.info(f"💡 {d}")

    # Tab 4: Competitor Gaps
    with tabs[3]:
        st.subheader("Unmet Needs & Competitor Gap Arbitrage")
        for g in insights.get("competitor_gaps", []):
            st.warning(f"🔍 {g}")

    # Tab 5: Shorts & Polls Studio
    with tabs[4]:
        st.subheader("Viral Shorts & Community Poll Concepts")
        sp = insights.get("shorts_polls", {})
        
        st.markdown("#### 🎬 Ready-to-Shoot Shorts")
        for idx, short in enumerate(sp.get("shorts", []), 1):
            with st.expander(f"Short #{idx}: {short.get('hook', '')}", expanded=True):
                st.markdown(f"**Concept:** {short.get('core_concept', '')}")
                st.markdown(f"**Call to Action:** `{short.get('cta', '')}`")
                
        st.markdown("#### 📊 Community Poll Questions")
        for idx, poll in enumerate(sp.get("polls", []), 1):
            with st.expander(f"Poll #{idx}: {poll.get('question', '')}", expanded=True):
                for opt in poll.get("options", []):
                    st.write(f"▫️ {opt}")

    # Tab 6: Doubts & Myths
    with tabs[5]:
        st.subheader("Audience Misconceptions & Doubts")
        for doubt in insights.get("doubts_myths", []):
            st.error(f"❓ {doubt}")

    # Tab 7: Action Blueprint
    with tabs[6]:
        st.subheader("Strategic Next Steps for Growth")
        for step in insights.get("action_blueprint", []):
            st.success(f"✅ {step}")

    # Tab 8: Raw Comments
    with tabs[7]:
        st.subheader("Fetched Raw Comments Dataset")
        st.dataframe(pd.DataFrame(comments, columns=["Comment Text"]), use_container_width=True)