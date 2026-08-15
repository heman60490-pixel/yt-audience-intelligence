import os
import re
import json
import time
import pandas as pd
import streamlit as st
import altair as alt
from fpdf import FPDF
from dotenv import load_dotenv
from googleapiclient.discovery import build
from groq import Groq

# 1. Environment & Client Setup
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
YT_KEY = os.getenv("YOUTUBE_API_KEY")

groq_client = Groq(api_key=GROQ_KEY)
youtube = build("youtube", "v3", developerKey=YT_KEY)


def extract_video_id(url_or_id):
    """Extract YouTube Video ID from URL or raw string"""
    if "v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id.strip()


def extract_channel_handle_or_id(url_or_str):
    """Extract channel identifier from standard channel URLs"""
    val = url_or_str.strip()
    if "@" in val:
        match = re.search(r"@([a-zA-Z0-9_-]+)", val)
        if match:
            return f"@{match.group(1)}"
    elif "/channel/" in val:
        return val.split("/channel/")[1].split("/")[0].split("?")[0]
    return val


def get_channel_top_videos(channel_identifier, max_videos=5):
    """Fetch recent videos from a channel handle or ID"""
    try:
        # Search channel by handle/query
        search_req = youtube.search().list(
            q=channel_identifier,
            type="channel",
            part="id,snippet",
            maxResults=1
        )
        search_res = search_req.execute()
        if not search_res.get("items"):
            return None, []
        
        channel_id = search_res["items"][0]["id"]["channelId"]
        channel_title = search_res["items"][0]["snippet"]["title"]

        # Fetch channel's uploads playlist
        ch_req = youtube.channels().list(part="contentDetails", id=channel_id)
        ch_res = ch_req.execute()
        uploads_playlist = ch_res["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        # Fetch top N videos
        pl_req = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist,
            maxResults=max_videos
        )
        pl_res = pl_req.execute()

        videos = []
        for item in pl_res.get("items", []):
            videos.append({
                "video_id": item["contentDetails"]["videoId"],
                "title": item["snippet"]["title"],
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"]
            })
        return channel_title, videos
    except Exception as e:
        st.error(f"Channel Fetch Error: {e}")
        return None, []


def fetch_video_details(video_id):
    """Fetch video metadata and statistics"""
    try:
        req = youtube.videos().list(part="snippet,statistics", id=video_id)
        res = req.execute()
        if res.get("items"):
            item = res["items"][0]
            return {
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"],
                "views": item["statistics"].get("viewCount", "0"),
                "total_comments": item["statistics"].get("commentCount", "0"),
            }
    except Exception as e:
        st.error(f"Metadata Fetch Error: {e}")
    return None


def fetch_comments_paginated(video_id, max_count=150):
    """Fetch top relevance comments with pagination"""
    comments = []
    next_page_token = None

    try:
        while len(comments) < max_count:
            fetch_limit = min(100, max_count - len(comments))
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=fetch_limit,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance"
            )
            response = request.execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": snippet["authorDisplayName"],
                    "likes": snippet["likeCount"],
                    "text": snippet["textDisplay"]
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except Exception:
        pass

    return comments


def clean_comments_list(raw_comments, max_char_per_comment=180, max_total_items=120):
    """Filter spam/links, prioritize by likes, enforce token safety"""
    cleaned = []
    sorted_comments = sorted(raw_comments, key=lambda x: x.get("likes", 0), reverse=True)

    for c in sorted_comments:
        text = c["text"].strip()
        if len(text.split()) >= 3 and not re.search(r"http[s]?://", text):
            truncated_text = text[:max_char_per_comment] + "..." if len(text) > max_char_per_comment else text
            cleaned.append(f"[{c['likes']} likes] {truncated_text}")
        
        if len(cleaned) >= max_total_items:
            break

    return cleaned


def analyze_with_groq(cleaned_comments, mode_description="Standard Analysis"):
    """Deep analysis using Groq Llama-3.3-70b with Semantic Clustering + Strategy Matrix"""
    combined_text = "\n".join(cleaned_comments)

    system_instruction = (
        "You are an elite YouTube Growth Intelligence AI. Return ONLY a strictly valid JSON object conforming to the schema. "
        "Do not output markdown code ticks or conversational preamble."
    )

    user_prompt = f"""
    Context & Objective: {mode_description}

    Comments Analyzed:
    {combined_text}

    Generate valid JSON:
    {{
      "semantic_clusters": [
        {{"category": "Confusion / Doubts", "count_percentage": 25, "summary": "Key technical areas confusing viewers"}},
        {{"category": "Content Requests & Ideas", "count_percentage": 30, "summary": "Direct topic demands"}},
        {{"category": "Critique & Flaws", "count_percentage": 15, "summary": "Points of disagreement or delivery issues"}},
        {{"category": "Praise & High Value", "count_percentage": 30, "summary": "Sections viewers loved most"}}
      ],
      "demands": [
        {{
          "topic": "Concise Topic Name",
          "percentage": 50,
          "details": "Explanation of audience demand.",
          "top_quote": "Exact high-liked quote"
        }}
      ],
      "competitor_gaps": [
        {{
          "gap_title": "Unmet Need / Flaw",
          "explanation": "What was missed or explained poorly.",
          "recommended_counter_video": "Exact counter angle to capture audience."
        }}
      ],
      "shorts_pipeline": [
        {{
          "hook_text": "3-second viral hook",
          "concept": "15-30s short outline",
          "source_comment": "Reference quote"
        }}
      ],
      "community_poll": {{
        "question": "Engaging poll question",
        "options": ["Option 1", "Option 2", "Option 3", "Option 4"]
      }},
      "faq_matrix": {{
        "unanswered_doubts": ["Direct recurring question from viewers"],
        "misconceptions_myths": ["Common false belief or misconception"]
      }},
      "action_plan": {{
        "next_title": "High-CTR clickable title",
        "thumbnail_hook": "Visual thumbnail description",
        "intro_hook": "First 30s retention script concept",
        "core_recommendation": "Main delivery advice"
      }}
    }}

    Rules:
    - In 'semantic_clusters' and 'demands', the count_percentage and percentage values MUST each sum to 100%.
    - Ensure 'competitor_gaps' has at least 2 clear actionable points.
    """

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=2500
    )

    return json.loads(response.choices[0].message.content)


def generate_markdown_report(title, subtitle, data, sample_count):
    """Formats all intelligence tabs into a comprehensive Markdown strategy doc"""
    plan = data.get("action_plan", {})
    poll = data.get("community_poll", {})
    faq = data.get("faq_matrix", {})

    report = f"""# YouTube Audience Intelligence & Strategy Report
**Subject:** {title}  
**Context:** {subtitle}  
**Engine:** Groq LPU (Llama 3.3 70B)  
**Sample Analyzed:** {sample_count} High-Engagement Comments  

---

## 1. Semantic Category Distribution
"""
    for c in data.get("semantic_clusters", []):
        report += f"- **{c.get('category', '')} ({c.get('count_percentage', 0)}%):** {c.get('summary', '')}\n"

    report += "\n---\n\n## 2. Executive Action Blueprint\n"
    report += f"* **Recommended High-CTR Title:** `{plan.get('next_title', 'N/A')}`\n"
    report += f"* **Visual Thumbnail Hook:** {plan.get('thumbnail_hook', 'N/A')}\n"
    report += f"* **First 30s Retention Hook:** {plan.get('intro_hook', 'N/A')}\n"
    report += f"* **Core Delivery Strategy:** {plan.get('core_recommendation', 'N/A')}\n"

    report += "\n---\n\n## 3. Audience Demand Breakdown\n"
    for d in data.get("demands", []):
        report += f"\n### • {d.get('topic', 'Topic')} ({d.get('percentage', 0)}% Demand)\n"
        report += f"- **Context:** {d.get('details', '')}\n"
        if d.get("top_quote"):
            report += f"- **Viewer Proof:** \"{d.get('top_quote')}\"\n"

    report += "\n---\n\n## 4. Competitor Gap Arbitrage Opportunities\n"
    for g in data.get("competitor_gaps", []):
        report += f"\n### Gap: {g.get('gap_title', 'Gap')}\n"
        report += f"- **What Was Missed:** {g.get('explanation', '')}\n"
        report += f"- **Counter Video Play:** {g.get('recommended_counter_video', '')}\n"

    report += "\n---\n\n## 5. Viral Shorts Concepts\n"
    for idx, s in enumerate(data.get("shorts_pipeline", []), 1):
        report += f"\n### Short #{idx}: \"{s.get('hook_text', '')}\"\n"
        report += f"- **Outline:** {s.get('concept', '')}\n"

    return report


def generate_pdf_report(title, subtitle, data, sample_count):
    """Creates a structured, robust PDF report using FPDF2"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    usable_w = pdf.epw

    def clean_text(txt):
        if not txt:
            return ""
        return str(txt).encode("latin-1", "replace").decode("latin-1").replace("?", " ")

    def write_section_heading(t):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(22, 101, 192)
        pdf.set_x(15)
        pdf.multi_cell(usable_w, 7, clean_text(t))
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(33, 37, 41)
    pdf.multi_cell(usable_w, 8, "YouTube Audience Intelligence & Strategy Report")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(usable_w, 5, clean_text(f"Target: {title} | Context: {subtitle}"))
    pdf.multi_cell(usable_w, 5, clean_text(f"Sample: {sample_count} comments | Engine: Groq Llama 3.3 70B"))
    
    pdf.ln(2)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)

    # Action Blueprint
    write_section_heading("1. Executive Action Blueprint")
    plan = data.get("action_plan", {})
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(15)
    pdf.multi_cell(usable_w, 5, "Recommended Title:")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_x(15)
    pdf.multi_cell(usable_w, 5, clean_text(plan.get("next_title", "N/A")))
    pdf.ln(1)

    # Demands
    write_section_heading("2. Audience Topic Demands")
    for d in data.get("demands", []):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_x(15)
        pdf.multi_cell(usable_w, 5, clean_text(f"- {d.get('topic', '')} ({d.get('percentage', 0)}% Demand)"))
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_x(15)
        pdf.multi_cell(usable_w, 4.5, clean_text(f"Details: {d.get('details', '')}"))
        pdf.ln(1.5)

    # Gaps
    write_section_heading("3. Gap Arbitrage Opportunities")
    for g in data.get("competitor_gaps", []):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_x(15)
        pdf.multi_cell(usable_w, 5, clean_text(f"- Gap: {g.get('gap_title', '')}"))
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_x(15)
        pdf.multi_cell(usable_w, 4.5, clean_text(f"Angle: {g.get('recommended_counter_video', '')}"))
        pdf.ln(1.5)

    return bytes(pdf.output())


# --- Streamlit UI Layout ---
st.set_page_config(
    page_title="Audience Intelligence & Strategy Suite",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .card-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .quote-box {
        background: #161b22;
        border-left: 3px solid #f0883e;
        padding: 8px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.9em;
        color: #e6edf3;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ YouTube Audience Intelligence Suite")
st.caption("Powered by Groq Llama-3.3-70B • Semantic Clustering, Channel-Level Mining, Gap Arbitrage & Strategy Generation.")

# Input Controls
col_input, col_mode, col_btn = st.columns([4.2, 2.2, 1.2])

with col_input:
    target_input = st.text_input("Enter YouTube Video URL or Channel Handle/URL:", placeholder="https://youtube.com/watch?v=... or @ChannelHandle", label_visibility="collapsed")

with col_mode:
    analysis_mode = st.selectbox(
        "Mode:",
        ["🎯 Single Video (Own Channel)", "⚔️ Single Video (Competitor Gap)", "📺 Channel-Level (Multi-Video Scan)"],
        label_visibility="collapsed"
    )

with col_btn:
    analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

if analyze_btn and target_input:
    raw_comments = []
    display_title = ""
    display_subtitle = ""
    display_thumb = None

    if "Channel-Level" in analysis_mode:
        channel_handle = extract_channel_handle_or_id(target_input)
        with st.spinner(f"Fetching latest videos from {channel_handle}..."):
            ch_name, videos = get_channel_top_videos(channel_handle, max_videos=5)

        if not videos:
            st.error("Channel not found or has no uploaded videos.")
        else:
            display_title = f"Channel Scan: {ch_name}"
            display_subtitle = f"Aggregated across {len(videos)} recent videos"
            display_thumb = videos[0]["thumbnail"]

            with st.spinner(f"Mining comments across {len(videos)} videos..."):
                for v in videos:
                    v_comments = fetch_comments_paginated(v["video_id"], max_count=50)
                    raw_comments.extend(v_comments)
    else:
        vid_id = extract_video_id(target_input)
        with st.spinner("Fetching video metrics..."):
            meta = fetch_video_details(vid_id)

        if not meta:
            st.error("Invalid Video URL or unable to retrieve metadata.")
        else:
            display_title = meta["title"]
            display_subtitle = f"Channel: {meta['channel']} | Mode: {analysis_mode}"
            display_thumb = meta["thumbnail"]
            with st.spinner("Fetching high-engagement comments..."):
                raw_comments = fetch_comments_paginated(vid_id, max_count=180)

    if raw_comments:
        cleaned_list = clean_comments_list(raw_comments, max_char_per_comment=180, max_total_items=120)

        # Header Info Banner
        st.divider()
        v1, v2, v3 = st.columns([1.2, 4, 1.5])
        with v1:
            if display_thumb:
                st.image(display_thumb, use_container_width=True)
        with v2:
            st.subheader(display_title)
            st.caption(display_subtitle)
        with v3:
            st.metric("Total Analyzed", f"{len(cleaned_list)} Comments")
        st.divider()

        # Groq Execution
        mode_prompt = f"Mode is {analysis_mode}. Perform deep market intelligence and gap extraction."
        data = None
        with st.spinner("⚡AI is clustering comments and synthesizing strategy in real-time..."):
            try:
                data = analyze_with_groq(cleaned_list, mode_description=mode_prompt)
            except Exception as err:
                st.error(f"Groq API Error: {err}")

        if data:
            # Download Row
            report_md = generate_markdown_report(display_title, display_subtitle, data, len(cleaned_list))
            report_pdf = generate_pdf_report(display_title, display_subtitle, data, len(cleaned_list))
            
            exp_space, exp_pdf, exp_md = st.columns([3.6, 1.8, 1.6])
            with exp_pdf:
                st.download_button("📄 Download PDF Report", data=report_pdf, file_name="YouTube_Strategy_Report.pdf", mime="application/pdf", type="primary", use_container_width=True)
            with exp_md:
                st.download_button("📝 Download (.md)", data=report_md, file_name="YouTube_Strategy_Report.md", mime="text/markdown", use_container_width=True)

            # Master Tabs
            tab_cluster, tab_demands, tab_gaps, tab_shorts, tab_faq, tab_plan, tab_raw = st.tabs([
                "📊 Semantic Clusters",
                "🎯 Demand Share (%)",
                "⚔️ Competitor Gaps",
                "📱 Shorts & Polls",
                "❓ Doubts & Myth Matrix",
                "🎬 Action Blueprint",
                "💬 Raw Comments"
            ])

            # TAB 0: SEMANTIC CLUSTERS
            with tab_cluster:
                st.subheader("Audience Sentiment & Intent Clustering")
                clusters = data.get("semantic_clusters", [])
                df_clusters = pd.DataFrame(clusters)

                if not df_clusters.empty and "count_percentage" in df_clusters.columns:
                    c_chart = alt.Chart(df_clusters).mark_arc(innerRadius=50).encode(
                        theta=alt.Theta(field="count_percentage", type="quantitative"),
                        color=alt.Color(field="category", type="nominal"),
                        tooltip=["category", "count_percentage", "summary"]
                    ).properties(height=260)
                    st.altair_chart(c_chart, use_container_width=True)

                    for cl in clusters:
                        st.markdown(f"""
                        <div class="card-box" style="border-left: 4px solid #8957e5;">
                            <h4 style="margin: 0; color: #d2a8ff;">{cl.get('category', '')} — {cl.get('count_percentage', 0)}%</h4>
                            <p style="margin: 4px 0 0 0; color: #c9d1d9;">{cl.get('summary', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # TAB 1: DEMANDS
            with tab_demands:
                st.subheader("Topic Demand Breakdown")
                demands_list = data.get("demands", [])
                df_demands = pd.DataFrame(demands_list)

                if not df_demands.empty and "percentage" in df_demands.columns:
                    chart = alt.Chart(df_demands).mark_bar(cornerRadiusEnd=4, color="#388bfd").encode(
                        x=alt.X('percentage:Q', title='Demand Share (%)', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('topic:N', sort='-x', title='Topic'),
                        tooltip=['topic', 'percentage', 'details']
                    ).properties(height=240)
                    st.altair_chart(chart, use_container_width=True)

                    for item in demands_list:
                        pct = item.get("percentage", 0)
                        topic = item.get("topic", "")
                        details = item.get("details", "")
                        quote = item.get("top_quote", "")
                        quote_html = f'<div class="quote-box">💬 <b>Proof:</b> "{quote}"</div>' if quote else ""

                        st.markdown(f"""
                        <div class="card-box" style="border-left: 4px solid #238636;">
                            <h4 style="margin: 0 0 6px 0; color: #58a6ff;">{topic} — <span style="color: #3fb950;">{pct}% Demand</span></h4>
                            <p style="margin: 0; color: #c9d1d9;">{details}</p>
                            {quote_html}
                        </div>
                        """, unsafe_allow_html=True)

            # TAB 2: COMPETITOR GAPS
            with tab_gaps:
                st.subheader("⚔️ Content Gap Arbitrage")
                for gap in data.get("competitor_gaps", []):
                    st.markdown(f"""
                    <div class="card-box" style="border-left: 4px solid #da3633;">
                        <h4 style="margin: 0 0 6px 0; color: #f85149;">⚠️ Gap: {gap.get('gap_title', '')}</h4>
                        <p style="color: #c9d1d9;"><b>Audience Issue:</b> {gap.get('explanation', '')}</p>
                        <div style="background: #161b22; padding: 10px; border-radius: 4px; border: 1px dashed #30363d; margin-top: 8px;">
                            🎯 <b>Recommended Counter Video:</b> <span style="color: #58a6ff;">{gap.get('recommended_counter_video', '')}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # TAB 3: SHORTS & POLLS
            with tab_shorts:
                st.subheader("📱 Viral Shorts / Reels Concepts")
                for s in data.get("shorts_pipeline", []):
                    st.markdown(f"""
                    <div class="card-box" style="border-left: 4px solid #a371f7;">
                        <h4 style="margin: 0 0 6px 0; color: #d2a8ff;">⚡ Hook: "{s.get('hook_text', '')}"</h4>
                        <p style="margin: 4px 0; color: #c9d1d9;"><b>Outline:</b> {s.get('concept', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                st.subheader("📊 Community Tab Engagement Poll")
                poll = data.get("community_poll", {})
                st.markdown(f"**Question:** `{poll.get('question', 'N/A')}`")
                for idx, opt in enumerate(poll.get("options", []), 1):
                    st.markdown(f"• **Option {idx}:** {opt}")

            # TAB 4: DOUBTS & FAQS
            with tab_faq:
                c_d, c_m = st.columns(2)
                faq = data.get("faq_matrix", {})
                with c_d:
                    st.subheader("❓ Top Unanswered Questions")
                    for d in faq.get("unanswered_doubts", []):
                        st.info(d, icon="💬")
                with c_m:
                    st.subheader("🚫 Myths & Misconceptions")
                    for m in faq.get("misconceptions_myths", []):
                        st.error(m, icon="❌")

            # TAB 5: ACTION BLUEPRINT
            with tab_plan:
                st.subheader("🎬 Execution Blueprint")
                plan = data.get("action_plan", {})
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown(f"**Recommended Title:**\n### `{plan.get('next_title', 'N/A')}`")
                    st.markdown(f"**🖼️ Thumbnail Concept:**\n{plan.get('thumbnail_hook', 'N/A')}")
                with p2:
                    st.markdown(f"**⏱️ First 30s Hook:**\n{plan.get('intro_hook', 'N/A')}")
                    st.markdown(f"**📦 Delivery Advice:**\n{plan.get('core_recommendation', 'N/A')}")

            # TAB 6: RAW COMMENTS
            with tab_raw:
                st.dataframe(pd.DataFrame(raw_comments).sort_values(by="likes", ascending=False), use_container_width=True)