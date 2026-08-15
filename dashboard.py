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
# 1. Page Configuration & SaaS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="YouTube Audience Demand & Doubt Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

custom_css = """
<style>
    /* Complete suppression of default Streamlit Cloud chrome & badges */
    #MainMenu, header, footer, [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"], div[class*="viewerBadge"], [data-testid="stViewerBadge"] {display: none !important;}
    div[class*="profile"], div[data-testid="stBottomBlockContainer"], [data-testid="stToolbar"] {display: none !important;}
    [class^="FloatingApp"], [class*="viewer_badge"], div[data-testid="stToolbarActions"] {display: none !important;}
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 94% !important;
    }
    
    /* SaaS Metric Boxes */
    .metric-card {
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
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Evidence Quote Styling */
    .evidence-container {
        background-color: #0f172a;
        border-left: 3px solid #38bdf8;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 10px;
    }
    .evidence-tag {
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 3px;
    }
    .evidence-text {
        font-size: 0.88rem;
        color: #cbd5e1;
        font-style: italic;
    }
    
    /* Content Opportunity Pitch Box */
    .pitch-card {
        background-color: #161b26;
        border: 1px solid #283347;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
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
# 3. Helpers, Noise Filtering & PDF Generator
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

def filter_high_intent_comments(comments: list) -> list:
    """Filter out generic 1-2 word praise and emoji-only spam to maximize LLM signal depth."""
    high_intent = []
    generic_patterns = [
        r"^(nice|good|great|super|mast|osm|op|wow|first|love you|sir|bhai|hello)\b.*",
        r"^(❤️|🔥|👏|😍|👍|\s)+$"
    ]
    
    for c in comments:
        text = c.strip()
        # Drop very short comments (< 15 characters) unless they contain question marks
        if len(text) < 15 and "?" not in text:
            continue
        
        # Check against low-signal patterns
        is_generic = False
        for p in generic_patterns:
            if re.match(p, text, re.IGNORECASE) and len(text.split()) <= 3:
                is_generic = True
                break
        
        if not is_generic:
            high_intent.append(text)
            
    # Fallback to original comments if filtering was too aggressive
    return high_intent if len(high_intent) >= 15 else comments

def generate_pdf_report(video_title, channel_title, kpis, demands, doubts, pitches):
    try:
        pdf = FPDF(format="A4", unit="mm")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        epw = pdf.epw
        
        # Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(epw, 10, "Audience Demand & Doubt Intelligence Brief", ln=True, align="C")
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(epw, 5, f"Target Video: {clean_pdf_text(video_title)[:80]}", ln=True, align="C")
        pdf.cell(epw, 5, f"Channel: {clean_pdf_text(channel_title)}", ln=True, align="C")
        pdf.ln(5)
        
        # Section 1: Executive Pulse
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "1. Audience Sentiment & Signal Ratio", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(epw, 6, f"- Net Sentiment: {kpis.get('sentiment_score', 'N/A')}% Positive", ln=True)
        pdf.cell(epw, 6, f"- Dominant Intent: {clean_pdf_text(kpis.get('dominant_intent', 'N/A'))}", ln=True)
        pdf.multi_cell(epw, 6, f"- Executive Summary: {clean_pdf_text(kpis.get('core_summary', 'N/A'))}")
        pdf.ln(4)
        
        # Section 2: Top Demands
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "2. High-Frequency Audience Demands", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for d in demands:
            vol = d.get('volume_percentage', 0)
            topic = clean_pdf_text(d.get('topic', ''))
            desc = clean_pdf_text(d.get('description', ''))
            pdf.multi_cell(epw, 6, f"- [{vol}% Demand] {topic}: {desc}")
            if d.get("quotes"):
                pdf.set_font("Helvetica", "I", 9)
                for q in d.get("quotes", [])[:1]:
                    pdf.multi_cell(epw, 5, f"  Evidence Quote: \"{clean_pdf_text(q)}\"")
                pdf.set_font("Helvetica", "", 10)
        pdf.ln(4)
        
        # Section 3: Critical Doubts
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "3. Critical Viewer Doubts & Knowledge Gaps", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for idx, doubt in enumerate(doubts, 1):
            q_text = clean_pdf_text(doubt.get('doubt', ''))
            sol = clean_pdf_text(doubt.get('creator_solution', ''))
            pdf.multi_cell(epw, 6, f"{idx}. Doubt: {q_text}")
            pdf.multi_cell(epw, 6, f"   Recommended Action: {sol}")
        pdf.ln(4)
        
        # Section 4: Next Video Pitches
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(epw, 7, "4. Recommended Next Video Blueprints", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for idx, pitch in enumerate(pitches, 1):
            title = clean_pdf_text(pitch.get('title', ''))
            hook = clean_pdf_text(pitch.get('hook_angle', ''))
            pdf.multi_cell(epw, 6, f"{idx}. Title: {title}")
            pdf.multi_cell(epw, 6, f"   Hook Angle: {hook}")
            
        return bytes(pdf.output())
    except Exception:
        return b""

# ---------------------------------------------------------
# 4. YouTube API Ingestion
# ---------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_youtube_comments(video_id: str, api_key: str, max_comments: int = 250):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        
        vid_res = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        if not vid_res.get("items"):
            return None, None, None
            
        video_meta = vid_res["items"][0]["snippet"]
        video_stats = vid_res["items"][0]["statistics"]
        
        raw_comments = []
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments),
            order="relevance",
            textFormat="plainText"
        )
        while req and len(raw_comments) < max_comments:
            res = req.execute()
            for item in res.get("items", []):
                t_comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                raw_comments.append(t_comment)
            req = youtube.commentThreads().list_next(req, res)
            
        return video_meta, video_stats, raw_comments
    except HttpError as e:
        if e.resp.status == 403:
            st.error("⚠️ YouTube API Quota exhausted for today or invalid key.")
        else:
            st.error(f"⚠️ YouTube API Error: {str(e)}")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error fetching comments: {str(e)}")
        return None, None, None

# ---------------------------------------------------------
# 5. Groq Precision Demand & Doubt Engine
# ---------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def run_precision_intelligence(comments: list, video_title: str) -> dict:
    prompt = f"""
    You are an Elite Audience Research Analyst and YouTube Algorithm Consultant.
    Perform deep mathematical demand and doubt extraction on these {len(comments)} high-intent comments for: "{video_title}".
    Context includes Hinglish, Hindi in Roman/Devanagari, and English idioms.

    CRITICAL RULES:
    1. Demand Volume %: Provide realistic percentage breakdown estimating audience interest share.
    2. Evidence Verbatim Quotes: For every demand and doubt, extract EXACT verbatim quotes directly from the comments as solid proof.
    3. Next Video Pitch: Provide complete high-CTR blueprints designed to pull guaranteed traffic from current video viewers.

    Return strict raw JSON with this exact schema:
    {{
      "kpis": {{
        "sentiment_score": (int 0-100),
        "dominant_intent": (string, e.g. "Seeking Practical Implementation Frameworks"),
        "core_summary": (concise 1-2 sentence high-impact audience consensus)
      }},
      "demand_breakdown": [
        {{
          "topic": (string topic title),
          "volume_percentage": (int estimated share of demands),
          "description": (clear explanation of what is needed),
          "quotes": [(verbatim quote 1), (verbatim quote 2)]
        }}
      ],
      "critical_doubts": [
        {{
          "doubt": (specific confusion or unanswered question),
          "evidence_quote": (exact verbatim quote),
          "creator_solution": (exact advice on how the creator should resolve it in content)
        }}
      ],
      "next_video_pitches": [
        {{
          "title": (high CTR YouTube video title based on audience demand),
          "hook_angle": (opening 15-second hook to retention-trap viewers),
          "core_points": [(bullet 1), (bullet 2), (bullet 3)],
          "demand_evidence": (exact viewer quote justifying this video)
        }}
      ]
    }}

    Comments Data:
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

def build_markdown_summary(video_title, kpis, demands, doubts, pitches):
    """Build a clean Markdown report for instant 1-click clipboard copying."""
    md = f"# 🎯 Audience Demand & Intelligence Brief\n"
    md += f"**Video:** {video_title}\n\n"
    md += f"### 📊 Key Signals\n"
    md += f"- **Sentiment Index:** {kpis.get('sentiment_score', 'N/A')}% Positive\n"
    md += f"- **Dominant Intent:** {kpis.get('dominant_intent', 'N/A')}\n"
    md += f"- **Consensus:** {kpis.get('core_summary', 'N/A')}\n\n"
    
    md += f"### 💡 Top Audience Demands (With Evidence)\n"
    for d in demands:
        md += f"- **[{d.get('volume_percentage')}% Demand] {d.get('topic')}**: {d.get('description')}\n"
        for q in d.get("quotes", [])[:1]:
            md += f"  > *\"{q}\"*\n"
    md += "\n"
    
    md += f"### ❓ Critical Viewer Doubts\n"
    for idx, doubt in enumerate(doubts, 1):
        md += f"{idx}. **Doubt:** {doubt.get('doubt')}\n"
        md += f"   > *\"{doubt.get('evidence_quote')}\"*\n"
        md += f"   **Action:** {doubt.get('creator_solution')}\n\n"
        
    md += f"### 🚀 Next High-ROI Video Blueprints\n"
    for idx, p in enumerate(pitches, 1):
        md += f"{idx}. **Title:** {p.get('title')}\n"
        md += f"   - **Hook:** {p.get('hook_angle')}\n"
        md += f"   - **Evidence:** *\"{p.get('demand_evidence')}\"*\n\n"
        
    return md

# ---------------------------------------------------------
# 6. Header & Input UI
# ---------------------------------------------------------
st.markdown("### 🎯 YouTube Audience Demand & Doubt Engine")
st.caption("Precision Evidence Mining • High-Intent Signal Filtering • Guaranteed-Traffic Video Pitches")

col_url, col_btn = st.columns([7, 2])
with col_url:
    yt_input_url = st.text_input(
        "URL",
        placeholder="Paste YouTube Video URL (e.g. https://youtu.be/...)",
        label_visibility="collapsed"
    )
with col_btn:
    analyze_pressed = st.button("🚀 Mine Audience Demands", use_container_width=True, type="primary")

# ---------------------------------------------------------
# 7. Pipeline Execution
# ---------------------------------------------------------
if analyze_pressed and yt_input_url:
    vid_id = extract_video_id(yt_input_url)
    if not vid_id:
        st.error("Please enter a valid YouTube video URL.")
        st.stop()
        
    with st.spinner("Fetching comments, removing low-signal spam & extracting evidence..."):
        meta, stats, raw_comments = fetch_youtube_comments(vid_id, yt_key)
        
        if not meta or not raw_comments:
            st.stop()
            
        filtered_comments = filter_high_intent_comments(raw_comments)
        insights = run_precision_intelligence(filtered_comments, meta.get("title", ""))
        
        st.session_state["pipeline_data"] = {
            "meta": meta,
            "stats": stats,
            "raw_comments": raw_comments,
            "filtered_count": len(filtered_comments),
            "insights": insights
        }

# ---------------------------------------------------------
# 8. Precision Output Dashboard
# ---------------------------------------------------------
if "pipeline_data" in st.session_state:
    data_state = st.session_state["pipeline_data"]
    meta = data_state["meta"]
    stats = data_state["stats"]
    raw_comments = data_state["raw_comments"]
    insights = data_state["insights"]
    kpis = insights.get("kpis", {})
    demands = insights.get("demand_breakdown", [])
    doubts = insights.get("critical_doubts", [])
    pitches = insights.get("next_video_pitches", [])

    st.markdown("---")

    # Video Overview Card
    c1, c2, c3 = st.columns([2, 5.5, 2.5])
    with c1:
        st.image(meta["thumbnails"]["high"]["url"], use_container_width=True)
    with c2:
        st.subheader(meta.get("title", ""))
        st.caption(f"Channel: **{meta.get('channelTitle', '')}** | Views: **{int(stats.get('viewCount', 0)):,}** | High-Intent Signals: **{data_state['filtered_count']}/{len(raw_comments)}**")
        st.info(f"💡 **Audience Consensus:** {kpis.get('core_summary', '')}")
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Audience Sentiment</div>
            <div class="metric-val">{kpis.get('sentiment_score', 85)}% Positive</div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">{kpis.get('dominant_intent', 'High Intent')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3 Focused Tabs
    tab_demands, tab_doubts, tab_pitches = st.tabs([
        "💡 High-Volume Audience Demands",
        "❓ Critical Doubts & Friction",
        "🚀 Next Video Pitch Blueprints"
    ])

    # Tab 1: Demands with Volume % and Quotes
    with tab_demands:
        col_pie, col_demands = st.columns([1, 1])
        
        with col_pie:
            if demands:
                df_d = pd.DataFrame(demands)
                fig = px.pie(
                    df_d, names="topic", values="volume_percentage", hole=0.6,
                    color_discrete_sequence=["#38bdf8", "#00d2ff", "#22c55e", "#f59e0b"]
                )
                fig.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
        with col_demands:
            st.markdown("#### Validated Demand Signals")
            for item in demands:
                with st.container():
                    st.markdown(f"**{item.get('topic')}** — `{item.get('volume_percentage')}% Demand Share`")
                    st.write(item.get("description", ""))
                    for q in item.get("quotes", []):
                        st.markdown(f"""
                        <div class="evidence-container">
                            <div class="evidence-tag">💬 Real Viewer Quote</div>
                            <div class="evidence-text">"{q}"</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Tab 2: Doubts & Friction Matrix
    with tab_doubts:
        st.markdown("#### Unanswered Doubts & Confusion in Audience Mind")
        for idx, d in enumerate(doubts, 1):
            with st.expander(f"Doubt #{idx}: {d.get('doubt', '')}", expanded=True):
                if d.get("evidence_quote"):
                    st.markdown(f"""
                    <div class="evidence-container">
                        <div class="evidence-tag">💬 Viewer Comment Evidence</div>
                        <div class="evidence-text">"{d.get('evidence_quote')}"</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.success(f"🛠️ **Recommended Creator Solution:** {d.get('creator_solution', '')}")

    # Tab 3: Actionable Next Video Pitches
    with tab_pitches:
        st.markdown("#### Next Video Blueprints with Guaranteed Audience Pull")
        for idx, pitch in enumerate(pitches, 1):
            st.markdown(f"""
            <div class="pitch-card">
                <h4 style="color: #38bdf8; margin-top: 0;">📌 Video Idea #{idx}: {pitch.get('title', '')}</h4>
                <p><strong>🎣 15-Second Hook:</strong> <em>"{pitch.get('hook_angle', '')}"</em></p>
                <div class="evidence-container">
                    <div class="evidence-tag">💬 Triggered By Audience Request</div>
                    <div class="evidence-text">"{pitch.get('demand_evidence', '')}"</div>
                </div>
                <strong>Key Content Breakdown:</strong>
                <ul>
                    {"".join([f"<li>{pt}</li>" for pt in pitch.get('core_points', [])])}
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # Bottom Actions & Fast Copy Bar
    st.markdown("---")
    st.markdown("#### 📋 1-Click Clipboard Export (Notion / Notes Ready)")
    md_content = build_markdown_summary(meta.get("title", ""), kpis, demands, doubts, pitches)
    st.code(md_content, language="markdown")

    d1, d2 = st.columns(2)
    with d1:
        pdf_bytes = generate_pdf_report(
            meta.get("title", ""),
            meta.get("channelTitle", ""),
            kpis,
            demands,
            doubts,
            pitches
        )
        if pdf_bytes:
            st.download_button("📥 Download Executive Summary PDF", data=pdf_bytes, file_name="Audience_Brief.pdf", mime="application/pdf", use_container_width=True)
    with d2:
        st.download_button("📄 Download Raw Comments (.txt)", data="\n\n".join(raw_comments), file_name="raw_comments.txt", mime="text/plain", use_container_width=True)