import os
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from groq import Groq

# ---------------------------------------------------------
# 1. Page Configuration & Aggressive UI Cleanup CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube Audience Intelligence Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

hide_streamlit_completely = """
<style>
    /* 1. Complete removal of Top Header, Hamburger Menu & Streamlit Brand */
    #MainMenu {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    
    /* 2. Target and completely remove the Bottom Status & Viewer Badges */
    [data-testid="stStatusWidget"] {display: none !important; visibility: hidden !important;}
    div[class*="viewerBadge"] {display: none !important; visibility: hidden !important;}
    [data-testid="stViewerBadge"] {display: none !important; visibility: hidden !important;}
    div[class*="profile"] {display: none !important; visibility: hidden !important;}
    div[data-testid="stBottomBlockContainer"] {display: none !important;}
    
    /* 3. Target bottom-right floating app badges, creator profiles, and toolbars */
    [data-testid="stToolbar"] {display: none !important; visibility: hidden !important;}
    [class^="FloatingApp"], [class*="viewer_badge"] {display: none !important;}
    button[title="View app in Streamlit Community Cloud"] {display: none !important;}
    div[data-testid="stToolbarActions"] {display: none !important;}
    
    /* 4. Streamlit layout padding optimization */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }
    
    /* 5. Custom Card Styling */
    .metric-card {
        background-color: #161a24;
        border: 1px solid #232a3b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
"""
st.markdown(hide_streamlit_completely, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Key Management (Secrets + Sidebar Fallback)
# ---------------------------------------------------------
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
yt_key = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))

if not groq_key or not yt_key:
    with st.sidebar:
        st.subheader("🔑 API Key Configuration")
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password", help="Enter Groq API Key")
        if not yt_key:
            yt_key = st.text_input("YouTube Data API Key", type="password", help="Enter YouTube API Key")

if not groq_key or not yt_key:
    st.warning("⚠️ Please provide both **GROQ_API_KEY** and **YOUTUBE_API_KEY** in Secrets or the Sidebar to proceed.")
    st.stop()

# Initialize Groq Client
client = Groq(api_key=groq_key)

# ---------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------
def extract_video_id(url: str) -> str:
    """Extract standard, share, or short YouTube video ID from URL."""
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_youtube_data(video_id: str, api_key: str, max_comments: int = 150):
    """Fetch video metadata and top-level comments with error handling."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        # 1. Fetch Video Details
        vid_response = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()
        
        if not vid_response.get("items"):
            return None, None
        
        video_details = vid_response["items"][0]["snippet"]
        
        # 2. Fetch Comments
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
            for item in res.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            req = youtube.commentThreads().list_next(req, res)
            
        return video_details, comments

    except HttpError as e:
        if e.resp.status == 403:
            st.error("⚠️ YouTube API Quota limit reached or invalid API Key.")
        else:
            st.error(f"⚠️ YouTube API Error: {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {str(e)}")
        return None, None

@st.cache_data(ttl=86400, show_spinner=False)
def run_groq_intelligence(comments: list, video_title: str) -> dict:
    """Analyze comments using Groq Llama 3.3 70B with Hinglish/Multilingual awareness."""
    prompt = f"""
    You are an expert Audience Intelligence Analyst.
    Analyze the following {len(comments)} YouTube comments for the video: "{video_title}".
    Note: Understand Hinglish, Hindi in Roman/Devanagari script, slang, and English contextually.

    Categorize and provide deep strategic synthesis in pure JSON format with the following keys:
    1. "clusters": List of objects with keys "category" (e.g. "Praise & High Value", "Content Requests & Ideas", "Critique & Flaws", "Confusion / Doubts"), "percentage" (number summing up to 100), "summary" (brief description).
    2. "demand_share": List of specific topics/questions viewers demand the most.
    3. "competitor_gaps": Unmet needs, missing explanations, or comparison angles.
    4. "shorts_polls": 3 ideas for YouTube Shorts and 2 Poll questions based on audience demand.
    5. "doubts_myths": Top doubts or misconceptions raised in comments.
    6. "action_blueprint": 4 concrete next steps for the creator.

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
# 4. Header & Input UI
# ---------------------------------------------------------
st.markdown("## ⚡ YouTube Audience Intelligence Suite")
st.caption("Powered by Groq Llama-3.3-70B • Semantic Clustering, Channel-Level Mining, Gap Arbitrage & Strategy Generation.")

col_url, col_mode, col_btn = st.columns([5, 3, 2])

with col_url:
    yt_input_url = st.text_input(
        "Video URL",
        placeholder="https://youtube.com/watch?v=... or https://youtu.be/...",
        label_visibility="collapsed"
    )

with col_mode:
    mode_selected = st.selectbox(
        "Analysis Mode",
        ["🎯 Single Video (Own Channel)", "⚔️ Competitor Analysis Mode"],
        label_visibility="collapsed"
    )

with col_btn:
    analyze_pressed = st.button("🚀 Analyze", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 5. Analysis Execution & Rendering
# ---------------------------------------------------------
if analyze_pressed and yt_input_url:
    video_id = extract_video_id(yt_input_url)
    if not video_id:
        st.error("Invalid YouTube URL. Please provide a valid video link.")
        st.stop()
        
    with st.spinner("Fetching comments & running deep neural clustering..."):
        video_details, comments = fetch_youtube_data(video_id, yt_key)
        
        if not video_details:
            st.stop()
            
        if not comments:
            st.warning("No comments found on this video or comments are disabled.")
            st.stop()
            
        insights = run_groq_intelligence(comments, video_details.get("title", ""))
        st.session_state["insights"] = insights
        st.session_state["video_details"] = video_details
        st.session_state["comments"] = comments

if "insights" in st.session_state:
    insights = st.session_state["insights"]
    vid_meta = st.session_state["video_details"]
    comments = st.session_state["comments"]

    st.markdown("---")
    
    # Video Overview Card
    c1, c2, c3 = st.columns([2, 5, 2])
    with c1:
        st.image(vid_meta["thumbnails"]["high"]["url"], use_container_width=True)
    with c2:
        st.subheader(vid_meta.get("title", ""))
        st.caption(f"Channel: **{vid_meta.get('channelTitle', '')}** | Mode: {mode_selected}")
    with c3:
        st.markdown(f"#### Total Analyzed")
        st.markdown(f"### **{len(comments)} Comments**")

    # Download Buttons Bar
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        st.download_button(
            "📄 Download Summary (.json)",
            data=json.dumps(insights, indent=2),
            file_name="audience_insights.json",
            mime="application/json",
            use_container_width=True
        )
    with btn_col2:
        st.download_button(
            "📝 Download Raw Comments (.txt)",
            data="\n\n".join(comments),
            file_name="comments_raw.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.markdown("---")
    
    # Navigation Tabs
    tabs = st.tabs([
        "📊 Semantic Clusters",
        "📈 Demand Share (%)",
        "⚔️ Competitor Gaps",
        "📱 Shorts & Polls",
        "❓ Doubts & Myth Matrix",
        "📋 Action Blueprint",
        "💬 Raw Comments"
    ])

    # Tab 1: Semantic Clusters & Donut Chart
    with tabs[0]:
        st.subheader("Audience Sentiment & Intent Clustering")
        clusters = insights.get("clusters", [])
        if clusters:
            df_clusters = pd.DataFrame(clusters)
            
            fig = px.pie(
                df_clusters,
                names="category",
                values="percentage",
                hole=0.55,
                color_discrete_sequence=["#FF4B4B", "#29B5E8", "#FFAA00", "#00CC96"]
            )
            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff")
            )
            st.plotly_chart(fig, use_container_width=True)

            for item in clusters:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{item.get('category')} — {item.get('percentage')}%</h4>
                    <p style="color: #a0aec0; margin: 0;">{item.get('summary')}</p>
                </div>
                """, unsafe_allow_html=True)

    # Tab 2: Demand Share
    with tabs[1]:
        st.subheader("Audience Demand & Topic Inquiries")
        for topic in insights.get("demand_share", []):
            st.info(f"💡 {topic}")

    # Tab 3: Competitor Gaps
    with tabs[2]:
        st.subheader("Content & Knowledge Gaps Identified")
        for gap in insights.get("competitor_gaps", []):
            st.warning(f"🔍 {gap}")

    # Tab 4: Shorts & Polls
    with tabs[3]:
        st.subheader("High-CTR Shorts & Poll Concepts")
        st.json(insights.get("shorts_polls", []))

    # Tab 5: Doubts & Myths
    with tabs[4]:
        st.subheader("Viewer Confusions & Misconceptions")
        for doubt in insights.get("doubts_myths", []):
            st.error(f"❓ {doubt}")

    # Tab 6: Action Blueprint
    with tabs[5]:
        st.subheader("Creator Next Steps Blueprint")
        for step in insights.get("action_blueprint", []):
            st.success(f"✅ {step}")

    # Tab 7: Raw Comments
    with tabs[6]:
        st.subheader("Fetched Raw Comments")
        st.dataframe(pd.DataFrame(comments, columns=["Comment Text"]), use_container_width=True)