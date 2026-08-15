import os
import re
import json
import unicodedata
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from groq import Groq

# ---------------------------------------------------------
# 1. Page Configuration & Minimalist Dark UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube Audience Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_styling = """
<style>
    /* Complete suppression of default Streamlit cloud chrome */
    #MainMenu, header, footer, [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"], div[class*="viewerBadge"], [data-testid="stViewerBadge"] {display: none !important;}
    div[class*="profile"], div[data-testid="stBottomBlockContainer"], [data-testid="stToolbar"] {display: none !important;}
    [class^="FloatingApp"], [class*="viewer_badge"], div[data-testid="stToolbarActions"] {display: none !important;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 92% !important;
    }
    
    /* Sleek KPI Cards */
    .kpi-card {
        background: #161b26;
        border: 1px solid #232d3f;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.8rem;
        color: #8b9bb4;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #00d2ff;
    }
    
    /* Content & Feedback Cards */
    .content-box {
        background-color: #161a24;
        border: 1px solid #232a3b;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(custom_styling, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. API Key Management
# ---------------------------------------------------------
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
yt_key = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))

if not groq_key or not yt_key:
    with st.sidebar:
        st.subheader("🔑 API Key Setup")
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password")
        if not yt_key:
            yt_key = st.text_input("YouTube API Key", type="password")

if not groq_key or not yt_key:
    st.warning("⚠️ Please provide **GROQ_API_KEY** and **YOUTUBE_API_KEY** to analyze videos.")
    st.stop()

client = Groq(api_key=groq_key)

# ---------------------------------------------------------
# 3. Helpers & Processing
# ---------------------------------------------------------
def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize('NFKD', str(text))
    ascii_text = norm.encode('ascii', 'ignore').decode('ascii')
    clean = re.sub(r'[\r\n\t]+', ' ', ascii_text)
    return re.sub(r'\s+', ' ', clean).strip()

def generate_pdf_report(video_title, channel_title, kpis, clusters, ideas):
    try:
        pdf = FPDF(format="A4", unit="mm")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        epw = pdf.epw
        
        # Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(epw, 10, "Creator Intelligence Report", ln=True, align="C")
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(epw, 5, f"Video: {clean_pdf_text(video_title)[:75]}...", ln=True, align="C")
        pdf.cell(epw, 5, f"Channel: {clean_pdf_text(channel_title)}", ln=True, align="C")
        pdf.ln(6)
        
        # KPIs
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "1. Audience Sentiment & Pulse", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(epw, 6, f"- Net Sentiment: {kpis.get('sentiment_score', 'N/A')}% Positive", ln=True)
        pdf.cell(epw, 6, f"- Audience Vibe: {clean_pdf_text(kpis.get('audience_vibe', 'N/A'))}", ln=True)
        pdf.ln(4)
        
        # Clusters
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "2. Top Audience Feedback", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for c in clusters:
            pdf.multi_cell(epw, 6, f"- {clean_pdf_text(c.get('category'))} ({c.get('percentage')}%): {clean_pdf_text(c.get('summary'))}")
        pdf.ln(4)
        
        # Ideas
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "3. Recommended Content Opportunities", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for idx, idea in enumerate(ideas, 1):
            pdf.multi_cell(epw, 6, f"{idx}. {clean_pdf_text(idea)}")
            
        return bytes(pdf.output())
    except Exception:
        return b""

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
            st.error("⚠️ YouTube API Quota exhausted for today.")
        else:
            st.error(f"⚠️ YouTube API Error: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {str(e)}")
        return None, None, None

@st.cache_data(ttl=86400, show_spinner=False)
def run_groq_intelligence(comments: list, video_title: str) -> dict:
    prompt = f"""
    You are a YouTube Content & Audience Growth Strategist.
    Analyze these {len(comments)} comments for the video: "{video_title}".
    Account for Hinglish, Hindi in Roman/Devanagari, and English context.

    Provide a clean, focused strategic response in JSON with:
    1. "kpis": {{
         "sentiment_score": (int 0-100),
         "audience_vibe": (short string, e.g. "Inspired & Enthusiastic"),
         "primary_takeaway": (short 1-sentence synthesis)
       }}
    2. "clusters": List of 3-4 objects with "category", "percentage", "summary".
    3. "content_opportunities": List of 3 actionable follow-up video or topic ideas based on demand.
    4. "shorts_concepts": List of 2 ready-to-shoot Shorts ideas with "hook" and "core_point".
    5. "top_questions_doubts": List of 3 real questions or confusions viewers asked.

    Comments Sample:
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

# ---------------------------------------------------------
# 4. Header UI
# ---------------------------------------------------------
st.markdown("### ⚡ YouTube Audience Intelligence")

col_url, col_btn = st.columns([6, 2])
with col_url:
    yt_input_url = st.text_input("URL", placeholder="Paste YouTube Video URL...", label_visibility="collapsed")
with col_btn:
    analyze_pressed = st.button("🚀 Analyze Video", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 5. Dashboard Output
# ---------------------------------------------------------
if analyze_pressed and yt_input_url:
    vid_id = extract_video_id(yt_input_url)
    if not vid_id:
        st.error("Please enter a valid YouTube link.")
        st.stop()
        
    with st.spinner("Analyzing audience feedback with AI..."):
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
    
    # Clean Video Overview Bar
    v1, v2, v3 = st.columns([2, 5, 2])
    with v1:
        st.image(vid_meta["thumbnails"]["high"]["url"], use_container_width=True)
    with v2:
        st.subheader(vid_meta.get("title", ""))
        st.caption(f"Channel: **{vid_meta.get('channelTitle', '')}** | Views: **{int(vid_stats.get('viewCount', 0)):,}**")
        st.info(f"💡 **Takeaway:** {kpis.get('primary_takeaway', '')}")
    with v3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Audience Sentiment</div>
            <div class="kpi-value">{kpis.get('sentiment_score', 85)}% Positive</div>
            <div style="font-size: 0.85rem; color: #8b9bb4; margin-top: 4px;">{kpis.get('audience_vibe', 'Supportive')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Simplified 3 Core Tabs
    tab_feedback, tab_ideas, tab_comments = st.tabs([
        "📊 Audience Sentiment & Feedback",
        "🚀 Next Video Ideas & Shorts",
        "💬 Viewer Questions & Comments"
    ])

    # Tab 1: Feedback & Clusters
    with tab_feedback:
        col_chart, col_list = st.columns([1, 1])
        clusters = insights.get("clusters", [])
        
        with col_chart:
            if clusters:
                df_c = pd.DataFrame(clusters)
                fig = px.pie(
                    df_c, names="category", values="percentage", hole=0.6,
                    color_discrete_sequence=["#00d2ff", "#29B5E8", "#FFAA00", "#FF4B4B"]
                )
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with col_list:
            st.markdown("#### What Viewers Are Saying")
            for item in clusters:
                st.markdown(f"""
                <div class="content-box">
                    <strong>{item.get('category')} ({item.get('percentage')}%)</strong>
                    <p style="color: #a0aec0; margin: 4px 0 0 0; font-size: 0.9rem;">{item.get('summary')}</p>
                </div>
                """, unsafe_allow_html=True)

    # Tab 2: Actionable Next Video Ideas & Shorts
    with tab_ideas:
        c_idea1, c_idea2 = st.columns(2)
        
        with c_idea1:
            st.markdown("#### 🎯 Next Long-Form Video Opportunities")
            for opp in insights.get("content_opportunities", []):
                st.success(f"💡 {opp}")
                
        with c_idea2:
            st.markdown("#### 🎬 Ready-to-Shoot Shorts Concepts")
            for s in insights.get("shorts_concepts", []):
                st.markdown(f"""
                <div class="content-box">
                    <strong style="color: #00d2ff;">Hook:</strong> "{s.get('hook', '')}"<br>
                    <span style="color: #cbd5e1; font-size: 0.9rem;"><strong>Angle:</strong> {s.get('core_point', '')}</span>
                </div>
                """, unsafe_allow_html=True)

    # Tab 3: Questions & Comments
    with tab_comments:
        st.markdown("#### ❓ Top Questions & Confusions from Viewers")
        for q in insights.get("top_questions_doubts", []):
            st.warning(f"🔍 {q}")
            
        st.markdown("---")
        with st.expander("📄 View All Fetched Comments"):
            st.dataframe(pd.DataFrame(comments, columns=["Comment Text"]), use_container_width=True)

    # Clean Download Options at Bottom
    st.markdown("---")
    down_col1, down_col2 = st.columns([1, 1])
    pdf_bytes = generate_pdf_report(
        vid_meta.get("title", ""),
        vid_meta.get("channelTitle", ""),
        kpis,
        insights.get("clusters", []),
        insights.get("content_opportunities", [])
    )
    with down_col1:
        if pdf_bytes:
            st.download_button("📥 Download Executive Summary PDF", data=pdf_bytes, file_name="Audience_Summary.pdf", mime="application/pdf", use_container_width=True)
    with down_col2:
        st.download_button("📄 Download Raw Comments (.txt)", data="\n\n".join(comments), file_name="comments.txt", mime="text/plain", use_container_width=True)