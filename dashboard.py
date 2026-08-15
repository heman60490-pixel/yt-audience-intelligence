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
# 1. Page Configuration & Modern SaaS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube Audience Intelligence Suite | SaaS Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_styling = """
<style>
    /* Complete suppression of default Streamlit Cloud Chrome */
    #MainMenu, header, footer, [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"], div[class*="viewerBadge"], [data-testid="stViewerBadge"] {display: none !important;}
    div[class*="profile"], div[data-testid="stBottomBlockContainer"], [data-testid="stToolbar"] {display: none !important;}
    [class^="FloatingApp"], [class*="viewer_badge"], div[data-testid="stToolbarActions"] {display: none !important;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 94% !important;
    }
    
    /* SaaS Metric Cards */
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
    
    /* Evidence & Quote Styling */
    .evidence-box {
        background-color: #121620;
        border-left: 3px solid #00d2ff;
        padding: 10px 14px;
        margin-top: 6px;
        margin-bottom: 12px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .quote-tag {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.78rem;
        text-transform: uppercase;
        margin-bottom: 2px;
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
        st.subheader("🔑 API Key Setup")
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password")
        if not yt_key:
            yt_key = st.text_input("YouTube API Key", type="password")

if not groq_key or not yt_key:
    st.warning("⚠️ Please provide **GROQ_API_KEY** and **YOUTUBE_API_KEY** in Secrets or Sidebar to proceed.")
    st.stop()

client = Groq(api_key=groq_key)

# ---------------------------------------------------------
# 3. Helpers & YouTube Extractors
# ---------------------------------------------------------
def clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    norm = unicodedata.normalize('NFKD', str(text))
    ascii_text = norm.encode('ascii', 'ignore').decode('ascii')
    clean = re.sub(r'[\r\n\t]+', ' ', ascii_text)
    return re.sub(r'\s+', ' ', clean).strip()

def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def extract_channel_identifier(input_str: str) -> tuple:
    """Detect if input is a handle (@channel), channel_id (UC...), or custom channel URL."""
    clean_str = input_str.strip()
    if "@" in clean_str:
        handle = re.search(r"@([\w.-]+)", clean_str)
        return ("handle", handle.group(1)) if handle else ("handle", clean_str.replace("@", ""))
    elif "channel/UC" in clean_str or clean_str.startswith("UC"):
        match = re.search(r"(UC[\w-]{22})", clean_str)
        return ("id", match.group(1)) if match else ("id", clean_str)
    return ("search", clean_str)

# ---------------------------------------------------------
# 4. Data Extraction Pipelines (Single Video + Channel Level)
# ---------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_single_video_data(video_id: str, api_key: str, max_comments: int = 150):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        vid_res = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        if not vid_res.get("items"):
            return None, None, None
        
        item = vid_res["items"][0]
        video_details = item["snippet"]
        video_stats = item["statistics"]
        
        comments = []
        req = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=min(100, max_comments), order="relevance", textFormat="plainText"
        )
        while req and len(comments) < max_comments:
            res = req.execute()
            for c_item in res.get("items", []):
                comments.append(c_item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            req = youtube.commentThreads().list_next(req, res)
            
        return video_details, video_stats, comments
    except Exception as e:
        st.error(f"Error fetching video data: {str(e)}")
        return None, None, None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_channel_batch_data(channel_query: str, api_key: str, num_videos: int = 5):
    """Fetch top recent videos from a channel and aggregate their comments."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        q_type, q_val = extract_channel_identifier(channel_query)
        
        channel_id = None
        if q_type == "handle":
            res = youtube.channels().list(part="snippet,contentDetails,statistics", forHandle=q_val).execute()
            if res.get("items"):
                channel_id = res["items"][0]["id"]
                channel_item = res["items"][0]
        elif q_type == "id":
            res = youtube.channels().list(part="snippet,contentDetails,statistics", id=q_val).execute()
            if res.get("items"):
                channel_id = res["items"][0]["id"]
                channel_item = res["items"][0]
                
        if not channel_id:
            # Fallback search
            search_res = youtube.search().list(part="snippet", q=q_val, type="channel", maxResults=1).execute()
            if search_res.get("items"):
                channel_id = search_res["items"][0]["snippet"]["channelId"]
                res = youtube.channels().list(part="snippet,contentDetails,statistics", id=channel_id).execute()
                channel_item = res["items"][0]
            else:
                return None, None, None

        uploads_playlist = channel_item["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Get last N videos from upload playlist
        playlist_res = youtube.playlistItems().list(
            part="snippet,contentDetails", playlistId=uploads_playlist, maxResults=num_videos
        ).execute()
        
        videos_meta = []
        all_comments = []
        for p_item in playlist_res.get("items", []):
            v_id = p_item["contentDetails"]["videoId"]
            v_title = p_item["snippet"]["title"]
            videos_meta.append({"id": v_id, "title": v_title})
            
            try:
                c_req = youtube.commentThreads().list(
                    part="snippet", videoId=v_id, maxResults=40, order="relevance", textFormat="plainText"
                ).execute()
                for c in c_req.get("items", []):
                    all_comments.append(c["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            except Exception:
                continue
                
        channel_meta = {
            "title": channel_item["snippet"]["title"],
            "thumbnail": channel_item["snippet"]["thumbnails"]["high"]["url"],
            "subscribers": channel_item["statistics"].get("subscriberCount", "N/A"),
            "videos_analyzed": len(videos_meta)
        }
        
        return channel_meta, videos_meta, all_comments
    except Exception as e:
        st.error(f"Error fetching channel data: {str(e)}")
        return None, None, None

# ---------------------------------------------------------
# 5. Evidence-Backed Groq Intelligence Core
# ---------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def run_groq_evidence_intelligence(comments: list, context_title: str, is_channel: bool = False) -> dict:
    prompt = f"""
    You are an Elite YouTube Audience Strategist and Data Analyst.
    Analyze the following {len(comments)} raw audience comments for: "{context_title}".
    Account for Hinglish, regional phrasing, slang, and core emotional resonance.

    CRITICAL REQUIREMENT: For every demand, friction point, and audience cluster, you MUST cite 1-2 REAL, EXACT verbatim quotes from the comments as evidence.

    Return pure JSON with this exact schema:
    {{
      "kpis": {{
        "sentiment_score": (int 0-100),
        "audience_vibe": (short punchy phrase, e.g. "Deeply Inspired & Seeking Action Steps"),
        "primary_takeaway": (high-impact 1-sentence synthesis)
      }},
      "clusters": [
        {{
          "category": (string),
          "percentage": (int),
          "summary": (concise overview),
          "evidence_quote": (exact verbatim quote from comments)
        }}
      ],
      "content_opportunities": [
        {{
          "title": (punchy high-CTR video title),
          "why_it_works": (why audience demands it),
          "evidence_quote": (real viewer quote demanding this)
        }}
      ],
      "shorts_studio": [
        {{
          "hook": (scroll-stopping opening hook in quotes),
          "core_script": (30-second structured script outline),
          "cta": (call to action)
        }}
      ],
      "audience_friction_doubts": [
        {{
          "doubt": (misconception, complaint or question),
          "evidence_quote": (exact quote representing this friction),
          "creator_solution": (how creator should address it)
        }}
      ]
    }}

    Comments Raw Data:
    {json.dumps(comments[:130], ensure_ascii=False)}

    Output STRICTLY pure JSON.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def generate_pdf_summary(title, subtitle, kpis, clusters, opps):
    try:
        pdf = FPDF(format="A4", unit="mm")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        epw = pdf.epw
        
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(epw, 10, "Audience Intelligence Executive Report", ln=True, align="C")
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(epw, 5, f"Target: {clean_pdf_text(title)[:80]}", ln=True, align="C")
        pdf.cell(epw, 5, f"Context: {clean_pdf_text(subtitle)}", ln=True, align="C")
        pdf.ln(6)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "1. Audience Sentiment & Core Pulse", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(epw, 6, f"- Net Sentiment: {kpis.get('sentiment_score', 'N/A')}% Positive", ln=True)
        pdf.cell(epw, 6, f"- Vibe: {clean_pdf_text(kpis.get('audience_vibe', 'N/A'))}", ln=True)
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "2. Evidence-Backed Feedback Clusters", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for c in clusters:
            pdf.multi_cell(epw, 6, f"- {clean_pdf_text(c.get('category'))} ({c.get('percentage')}%): {clean_pdf_text(c.get('summary'))}")
            if c.get("evidence_quote"):
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(epw, 5, f"  Quote: \"{clean_pdf_text(c.get('evidence_quote'))}\"")
                pdf.set_font("Helvetica", "", 10)
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "3. High-ROI Content Opportunities", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for idx, o in enumerate(opps, 1):
            pdf.multi_cell(epw, 6, f"{idx}. {clean_pdf_text(o.get('title'))} - {clean_pdf_text(o.get('why_it_works'))}")
            
        return bytes(pdf.output())
    except Exception:
        return b""

# ---------------------------------------------------------
# 6. Header UI & Mode Switcher
# ---------------------------------------------------------
st.markdown("### ⚡ YouTube Audience Intelligence Suite")

col_mode, col_input, col_btn = st.columns([2.5, 5.5, 2])

with col_mode:
    analysis_mode = st.selectbox(
        "Mode",
        ["🎯 Single Video Audit", "🌐 Channel Batch Mining (5 Videos)"],
        label_visibility="collapsed"
    )

with col_input:
    placeholder_txt = "Paste Video URL (e.g., https://youtu.be/...)" if "Single" in analysis_mode else "Paste Channel Handle or Link (e.g., @HubermanLab)"
    input_query = st.text_input("Query", placeholder=placeholder_txt, label_visibility="collapsed")

with col_btn:
    analyze_pressed = st.button("🚀 Run Intelligence", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 7. Pipeline Execution & State Caching
# ---------------------------------------------------------
if analyze_pressed and input_query:
    if "Single" in analysis_mode:
        vid_id = extract_video_id(input_query)
        if not vid_id:
            st.error("Invalid YouTube video URL.")
            st.stop()
        with st.spinner("Extracting video comments & mining evidence..."):
            meta, stats, comments = fetch_single_video_data(vid_id, yt_key)
            if not meta or not comments:
                st.stop()
            insights = run_groq_evidence_intelligence(comments, meta.get("title", ""), is_channel=False)
            st.session_state["data"] = {
                "type": "video",
                "meta": meta,
                "stats": stats,
                "comments": comments,
                "insights": insights
            }
    else:
        with st.spinner("Mining latest 5 videos from channel & extracting cross-audience trends..."):
            ch_meta, vids, comments = fetch_channel_batch_data(input_query, yt_key)
            if not ch_meta or not comments:
                st.error("Could not fetch channel. Verify channel handle/URL.")
                st.stop()
            insights = run_groq_evidence_intelligence(comments, f"Channel: {ch_meta['title']}", is_channel=True)
            st.session_state["data"] = {
                "type": "channel",
                "meta": ch_meta,
                "videos": vids,
                "comments": comments,
                "insights": insights
            }

# ---------------------------------------------------------
# 8. Production Dashboard Interface
# ---------------------------------------------------------
if "data" in st.session_state:
    data_state = st.session_state["data"]
    insights = data_state["insights"]
    kpis = insights.get("kpis", {})
    comments = data_state["comments"]
    
    st.markdown("---")

    # Overview Banner
    if data_state["type"] == "video":
        meta = data_state["meta"]
        stats = data_state["stats"]
        v1, v2, v3 = st.columns([2, 5.5, 2.5])
        with v1:
            st.image(meta["thumbnails"]["high"]["url"], use_container_width=True)
        with v2:
            st.subheader(meta.get("title", ""))
            st.caption(f"Channel: **{meta.get('channelTitle', '')}** | Views: **{int(stats.get('viewCount', 0)):,}** | Comments Analyzed: **{len(comments)}**")
            st.info(f"💡 **Executive Takeaway:** {kpis.get('primary_takeaway', '')}")
    else:
        ch_meta = data_state["meta"]
        v1, v2, v3 = st.columns([1.5, 6, 2.5])
        with v1:
            st.image(ch_meta["thumbnail"], use_container_width=True)
        with v2:
            st.subheader(f"Channel Audit: {ch_meta.get('title', '')}")
            st.caption(f"Batch Scope: **Last {ch_meta['videos_analyzed']} Uploads** | Subscribers: **{ch_meta['subscribers']}** | Total Comments: **{len(comments)}**")
            st.info(f"💡 **Channel-Wide Synthesis:** {kpis.get('primary_takeaway', '')}")

    with v3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Audience Sentiment Index</div>
            <div class="kpi-value">{kpis.get('sentiment_score', 85)}% Positive</div>
            <div style="font-size: 0.85rem; color: #8b9bb4; margin-top: 4px;">{kpis.get('audience_vibe', 'Supportive')}</div>
        </div>
        """, unsafe_allow_html=True)

    # Core Actionable Tabs
    tab_feedback, tab_ideas, tab_shorts, tab_doubts, tab_raw = st.tabs([
        "📊 Evidence-Backed Feedback",
        "💡 Next Video Blueprints",
        "🎬 Shorts Studio (1-Click Copy)",
        "❓ Friction & Doubts Matrix",
        "📄 Raw Comments Dataset"
    ])

    # Tab 1: Evidence-Backed Feedback & Pie Chart
    with tab_feedback:
        col_c1, col_c2 = st.columns([1, 1])
        clusters = insights.get("clusters", [])
        
        with col_c1:
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
                
        with col_c2:
            st.markdown("#### What Viewers Are Saying (With Evidence)")
            for item in clusters:
                with st.container():
                    st.markdown(f"**{item.get('category')} ({item.get('percentage')}%)**")
                    st.write(item.get("summary", ""))
                    if item.get("evidence_quote"):
                        st.markdown(f"""
                        <div class="evidence-box">
                            <div class="quote-tag">💬 Real Viewer Evidence Quote</div>
                            "{item.get('evidence_quote')}"
                        </div>
                        """, unsafe_allow_html=True)

    # Tab 2: High-ROI Content Opportunities with Quotes
    with tab_ideas:
        st.markdown("#### 🎯 Demand-Driven Next Long-Form Videos")
        for opp in insights.get("content_opportunities", []):
            with st.expander(f"📌 {opp.get('title', 'Video Idea')}", expanded=True):
                st.markdown(f"**Strategic Rationale:** {opp.get('why_it_works', '')}")
                if opp.get("evidence_quote"):
                    st.markdown(f"""
                    <div class="evidence-box">
                        <div class="quote-tag">💬 Prompted By Viewer Demand</div>
                        "{opp.get('evidence_quote')}"
                    </div>
                    """, unsafe_allow_html=True)

    # Tab 3: Shorts Studio with Native Copy Buttons
    with tab_shorts:
        st.markdown("#### 🎬 Ready-to-Produce Viral Shorts Concepts")
        shorts_list = insights.get("shorts_studio", [])
        for idx, s in enumerate(shorts_list, 1):
            with st.container():
                st.markdown(f"##### Short #{idx}: Hook Proposal")
                st.code(s.get("hook", ""), language="markdown")
                st.markdown(f"**Core Script Outline:**\n{s.get('core_script', '')}")
                st.caption(f"Suggested Call to Action: `{s.get('cta', '')}`")
                st.markdown("---")

    # Tab 4: Friction & Misconceptions
    with tab_doubts:
        st.markdown("#### 🔍 Viewer Objections, Confusions & Knowledge Gaps")
        for f in insights.get("audience_friction_doubts", []):
            with st.container():
                st.markdown(f"**Issue / Doubt:** `{f.get('doubt', '')}`")
                if f.get("evidence_quote"):
                    st.markdown(f"""
                    <div class="evidence-box">
                        <div class="quote-tag">💬 Exact Viewer Feedback</div>
                        "{f.get('evidence_quote')}"
                    </div>
                    """, unsafe_allow_html=True)
                st.success(f"💡 **Recommended Creator Action:** {f.get('creator_solution', '')}")
                st.markdown("<br>", unsafe_allow_html=True)

    # Tab 5: Raw Dataset
    with tab_raw:
        st.markdown("#### Analyzed Comments Dataset")
        st.dataframe(pd.DataFrame(comments, columns=["Comment Text"]), use_container_width=True)

    # Export Bar
    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        target_name = data_state["meta"].get("title", "Report")
        pdf_bytes = generate_pdf_summary(
            target_name,
            "Channel Batch" if data_state["type"] == "channel" else "Single Video",
            kpis,
            insights.get("clusters", []),
            insights.get("content_opportunities", [])
        )
        if pdf_bytes:
            st.download_button("📥 Download Executive Summary PDF", data=pdf_bytes, file_name="Audience_Report.pdf", mime="application/pdf", use_container_width=True)
    with d2:
        st.download_button("📄 Download Raw Comments (.txt)", data="\n\n".join(comments), file_name="comments_dataset.txt", mime="text/plain", use_container_width=True)