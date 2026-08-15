import os
import re
import json
import io
import unicodedata
import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from groq import Groq

# ---------------------------------------------------------
# 1. Page Configuration & Clean Dark SaaS Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube Comment Analyzer & Intelligence Suite",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_css = """
<style>
    /* Complete suppression of default Streamlit Cloud Chrome & Badges */
    #MainMenu, header, footer, [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"], div[class*="viewerBadge"], [data-testid="stViewerBadge"] {display: none !important;}
    div[class*="profile"], div[data-testid="stBottomBlockContainer"], [data-testid="stToolbar"] {display: none !important;}
    [class^="FloatingApp"], [class*="viewer_badge"], div[data-testid="stToolbarActions"] {display: none !important;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 94% !important;
    }
    
    /* SaaS Metric Cards */
    .metric-box {
        background: #141923;
        border: 1px solid #232d3f;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Evidence Quote Box */
    .quote-box {
        background-color: #0f172a;
        border-left: 3px solid #38bdf8;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        margin-top: 6px;
        margin-bottom: 12px;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .quote-label {
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Key Management (Secrets + Sidebar Fallback)
# ---------------------------------------------------------
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
yt_key = st.secrets.get("YOUTUBE_API_KEY", os.getenv("YOUTUBE_API_KEY", ""))

if not groq_key or not yt_key:
    with st.sidebar:
        st.subheader("🔑 API Key Setup")
        if not groq_key:
            groq_key = st.text_input("Groq API Key", type="password", help="Enter Groq API Key")
        if not yt_key:
            yt_key = st.text_input("YouTube Data API Key", type="password", help="Enter YouTube API Key")

if not groq_key or not yt_key:
    st.warning("⚠️ Please provide both **GROQ_API_KEY** and **YOUTUBE_API_KEY** in Secrets or Sidebar.")
    st.stop()

client = Groq(api_key=groq_key)

# ---------------------------------------------------------
# 3. Helpers & PDF Sanitization
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

def generate_pdf_report(video_title, channel_title, kpis, clusters, questions, ideas):
    try:
        pdf = FPDF(format="A4", unit="mm")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        epw = pdf.epw
        
        # Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(epw, 10, "YouTube Comment & Audience Intelligence Report", ln=True, align="C")
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(epw, 5, f"Video: {clean_pdf_text(video_title)[:80]}", ln=True, align="C")
        pdf.cell(epw, 5, f"Channel: {clean_pdf_text(channel_title)}", ln=True, align="C")
        pdf.ln(5)
        
        # Section 1: Overview
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "1. Sentiment & Audience Vibe", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(epw, 6, f"- Net Sentiment: {kpis.get('sentiment_score', 'N/A')}% Positive", ln=True)
        pdf.cell(epw, 6, f"- Dominant Emotion: {clean_pdf_text(kpis.get('dominant_emotion', 'N/A'))}", ln=True)
        pdf.multi_cell(epw, 6, f"- Key Synthesis: {clean_pdf_text(kpis.get('summary_takeaway', 'N/A'))}")
        pdf.ln(4)
        
        # Section 2: Feedback Clusters
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "2. Comment Themes & Breakdown", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for c in clusters:
            cat = clean_pdf_text(c.get('category', ''))
            pct = c.get('percentage', 0)
            summ = clean_pdf_text(c.get('summary', ''))
            pdf.multi_cell(epw, 6, f"- {cat} ({pct}%): {summ}")
            if c.get("evidence_quote"):
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(epw, 5, f"  Quote: \"{clean_pdf_text(c.get('evidence_quote'))}\"")
                pdf.set_font("Helvetica", "", 10)
        pdf.ln(4)
        
        # Section 3: Questions
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "3. Top Viewer Questions & Doubts", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for idx, q in enumerate(questions, 1):
            pdf.multi_cell(epw, 6, f"{idx}. {clean_pdf_text(q.get('question', ''))}")
        pdf.ln(4)
        
        # Section 4: Content Action Ideas
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "4. Demand-Driven Next Steps", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for idx, idea in enumerate(ideas, 1):
            pdf.multi_cell(epw, 6, f"{idx}. {clean_pdf_text(idea.get('title', ''))} - {clean_pdf_text(idea.get('rationale', ''))}")
            
        return bytes(pdf.output())
    except Exception:
        return b""

# ---------------------------------------------------------
# 4. YouTube Comment Data Extraction
# ---------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_comments_dataset(video_id: str, api_key: str, max_comments: int = 250):
    """Fetch video metadata and top relevant/latest comments dataset."""
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        # Video Details
        vid_res = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        if not vid_res.get("items"):
            return None, None, None
            
        video_meta = vid_res["items"][0]["snippet"]
        video_stats = vid_res["items"][0]["statistics"]
        
        # Fetch Comments with details (Author, Like Count, Date)
        comments_list = []
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments),
            order="relevance",
            textFormat="plainText"
        )
        while req and len(comments_list) < max_comments:
            res = req.execute()
            for item in res.get("items", []):
                t_comment = item["snippet"]["topLevelComment"]["snippet"]
                comments_list.append({
                    "author": t_comment.get("authorDisplayName", "User"),
                    "text": t_comment.get("textDisplay", ""),
                    "likes": t_comment.get("likeCount", 0),
                    "published_at": t_comment.get("publishedAt", "")[:10]
                })
            req = youtube.commentThreads().list_next(req, res)
            
        return video_meta, video_stats, comments_list
    except HttpError as e:
        if e.resp.status == 403:
            st.error("⚠️ YouTube API Quota limit reached for today or invalid key.")
        else:
            st.error(f"⚠️ YouTube API Error: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error fetching comments: {str(e)}")
        return None, None, None

# ---------------------------------------------------------
# 5. Groq Llama-3.3-70B Deep Comment Analysis
# ---------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def run_deep_comment_analysis(comments_records: list, video_title: str) -> dict:
    """Analyze comment records with multilingual context and extract deep intelligence."""
    clean_texts = [c["text"] for c in comments_records if c["text"].strip()]
    
    prompt = f"""
    You are an expert YouTube Audience & Comment Analyst.
    Perform a comprehensive, evidence-backed comment analysis for the video titled: "{video_title}".
    Account for Hinglish, regional phrasing, emotional subtext, slang, and core viewer sentiment.

    Analyze these {len(clean_texts)} viewer comments and provide strict raw JSON output with this schema:
    {{
      "kpis": {{
        "sentiment_score": (int 0-100 representing positive-to-constructive ratio),
        "dominant_emotion": (short phrase, e.g. "Inspired, Grateful & Curious"),
        "summary_takeaway": (concise 1-2 sentence core message of audience feedback)
      }},
      "themes_clusters": [
        {{
          "category": (string, e.g. "Key Value Learned", "Constructive Feedback", "Personal Stories", "Content Requests"),
          "percentage": (int percentage summing up to 100),
          "summary": (clear description of this theme),
          "evidence_quote": (exact verbatim quote from comments)
        }}
      ],
      "viewer_questions": [
        {{
          "question": (the specific question/doubt raised),
          "evidence_quote": (exact viewer comment containing this query),
          "recommended_response": (how the creator should address this)
        }}
      ],
      "content_opportunities": [
        {{
          "title": (suggested follow-up video or topic title),
          "rationale": (why viewers want this based on comments),
          "evidence_quote": (exact quote showing this demand)
        }}
      ],
      "shorts_ideas": [
        {{
          "hook": (scroll-stopping opening hook),
          "script_outline": (concise 30s point-by-point flow),
          "call_to_action": (suggested CTA)
        }}
      ]
    }}

    Comments Sample Dataset:
    {json.dumps(clean_texts[:130], ensure_ascii=False)}

    Output STRICTLY raw JSON.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# ---------------------------------------------------------
# 6. Header UI & Input Box
# ---------------------------------------------------------
st.markdown("### 💬 YouTube Comment Analyzer & Intelligence")
st.caption("AI-Powered Semantic Mining, Real Evidence Extraction & Creator Strategy Engine")

col_url, col_btn = st.columns([7, 2])
with col_url:
    yt_input_url = st.text_input(
        "Video URL",
        placeholder="Paste YouTube Video URL (e.g., https://youtu.be/... or https://youtube.com/watch?v=...)",
        label_visibility="collapsed"
    )
with col_btn:
    analyze_pressed = st.button("🚀 Analyze Comments", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 7. Pipeline Execution
# ---------------------------------------------------------
if analyze_pressed and yt_input_url:
    vid_id = extract_video_id(yt_input_url)
    if not vid_id:
        st.error("Please enter a valid YouTube video URL.")
        st.stop()
        
    with st.spinner("Extracting viewer comments & executing deep neural analysis..."):
        meta, stats, comments_data = fetch_comments_dataset(vid_id, yt_key)
        
        if not meta or not comments_data:
            st.stop()
            
        insights = run_deep_comment_analysis(comments_data, meta.get("title", ""))
        st.session_state["comment_intel"] = {
            "meta": meta,
            "stats": stats,
            "comments": comments_data,
            "insights": insights
        }

# ---------------------------------------------------------
# 8. Production Dashboard Interface
# ---------------------------------------------------------
if "comment_intel" in st.session_state:
    data_state = st.session_state["comment_intel"]
    meta = data_state["meta"]
    stats = data_state["stats"]
    comments_list = data_state["comments"]
    insights = data_state["insights"]
    kpis = insights.get("kpis", {})

    st.markdown("---")

    # Overview Banner
    c1, c2, c3 = st.columns([2, 5.5, 2.5])
    with c1:
        st.image(meta["thumbnails"]["high"]["url"], use_container_width=True)
    with c2:
        st.subheader(meta.get("title", ""))
        st.caption(f"Channel: **{meta.get('channelTitle', '')}** | Total Views: **{int(stats.get('viewCount', 0)):,}** | Comments Analyzed: **{len(comments_list)}**")
        st.info(f"💡 **Audience Synthesis:** {kpis.get('summary_takeaway', '')}")
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Audience Sentiment</div>
            <div class="metric-value">{kpis.get('sentiment_score', 85)}% Positive</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">{kpis.get('dominant_emotion', 'Inspired')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Analyzer Tabs
    t_themes, t_questions, t_ideas, t_shorts, t_raw = st.tabs([
        "📊 Comment Themes & Evidence",
        "❓ Viewer Questions & Doubts",
        "💡 Next Video Ideas",
        "🎬 Ready Shorts Scripts",
        "🔍 Search & Filter Raw Comments"
    ])

    # Tab 1: Themes & Evidence
    with t_themes:
        col_pie, col_themes = st.columns([1, 1])
        clusters = insights.get("themes_clusters", [])
        
        with col_pie:
            if clusters:
                df_c = pd.DataFrame(clusters)
                fig = px.pie(
                    df_c, names="category", values="percentage", hole=0.6,
                    color_discrete_sequence=["#38bdf8", "#22c55e", "#f59e0b", "#ef4444"]
                )
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_themes:
            st.markdown("#### What Viewers Are Saying")
            for item in clusters:
                with st.container():
                    st.markdown(f"**{item.get('category')} ({item.get('percentage')}%)**")
                    st.write(item.get("summary", ""))
                    if item.get("evidence_quote"):
                        st.markdown(f"""
                        <div class="quote-box">
                            <div class="quote-label">💬 Real Viewer Quote</div>
                            "{item.get('evidence_quote')}"
                        </div>
                        """, unsafe_allow_html=True)

    # Tab 2: Questions & Doubts Extractor
    with t_questions:
        st.markdown("#### ❓ Questions & Doubts Extracted From Comments")
        questions_list = insights.get("viewer_questions", [])
        if questions_list:
            for idx, q in enumerate(questions_list, 1):
                with st.expander(f"Question #{idx}: {q.get('question', '')}", expanded=True):
                    if q.get("evidence_quote"):
                        st.markdown(f"""
                        <div class="quote-box">
                            <div class="quote-label">💬 Viewer Comment</div>
                            "{q.get('evidence_quote')}"
                        </div>
                        """, unsafe_allow_html=True)
                    st.success(f"💡 **Recommended Creator Response / Video Angle:** {q.get('recommended_response', '')}")
        else:
            st.info("No prominent recurring doubts identified in this comment sample.")

    # Tab 3: Next Video Ideas
    with t_ideas:
        st.markdown("#### 🎯 Next High-Demand Video Opportunities")
        for idea in insights.get("content_opportunities", []):
            with st.expander(f"📌 {idea.get('title', '')}", expanded=True):
                st.markdown(f"**Audience Demand Rationale:** {idea.get('rationale', '')}")
                if idea.get("evidence_quote"):
                    st.markdown(f"""
                    <div class="quote-box">
                        <div class="quote-label">💬 Prompted By Viewer Demand</div>
                        "{idea.get('evidence_quote')}"
                    </div>
                    """, unsafe_allow_html=True)

    # Tab 4: Ready Shorts Scripts (with copy block)
    with t_shorts:
        st.markdown("#### 🎬 Ready-to-Shoot Shorts from Audience Feedback")
        for idx, s in enumerate(insights.get("shorts_ideas", []), 1):
            with st.container():
                st.markdown(f"##### Short #{idx}: Hook Proposal")
                st.code(s.get("hook", ""), language="markdown")
                st.markdown(f"**Script Flow:**\n{s.get('script_outline', '')}")
                st.caption(f"Call to Action: `{s.get('call_to_action', '')}`")
                st.markdown("---")

    # Tab 5: Interactive Searchable Raw Comments Table
    with t_raw:
        st.markdown("#### 🔍 Filter & Search Raw Comments Dataset")
        df_comments = pd.DataFrame(comments_list)
        
        search_term = st.text_input("Filter comments by keyword:", placeholder="Type a keyword to filter (e.g. overthinking, audio, question)...")
        if search_term:
            filtered_df = df_comments[df_comments["text"].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df_comments
            
        st.dataframe(
            filtered_df[["author", "likes", "published_at", "text"]],
            use_container_width=True,
            height=380
        )
        
        # Download Raw CSV
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Comments as CSV", data=csv_data, file_name="comments_data.csv", mime="text/csv")

    # Bottom Export Actions
    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        pdf_bytes = generate_pdf_report(
            meta.get("title", ""),
            meta.get("channelTitle", ""),
            kpis,
            insights.get("themes_clusters", []),
            insights.get("viewer_questions", []),
            insights.get("content_opportunities", [])
        )
        if pdf_bytes:
            st.download_button("📥 Download Executive Summary PDF", data=pdf_bytes, file_name="Comment_Intelligence_Report.pdf", mime="application/pdf", use_container_width=True)
    with d2:
        st.download_button("📄 Download Full JSON Analysis", data=json.dumps(insights, indent=2), file_name="comment_insights.json", mime="application/json", use_container_width=True)