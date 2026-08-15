import os
import re
from dotenv import load_dotenv
from google import genai
from googleapiclient.discovery import build

# 1. Load environment variables
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
YT_KEY = os.getenv("YOUTUBE_API_KEY")

# 2. Initialize API clients
ai_client = genai.Client(api_key=GEMINI_KEY)
youtube = build("youtube", "v3", developerKey=YT_KEY)


def extract_video_id(url_or_id):
    """Extract YouTube Video ID from URL or return raw ID"""
    if "v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id


def fetch_comments(video_id, max_count=100):
    """Fetch top comments from YouTube Data API"""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_count, 100),
            textFormat="plainText",
            order="relevance",
        )
        response = request.execute()
        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append(
                {
                    "text": snippet["textDisplay"],
                    "likes": snippet["likeCount"],
                }
            )
    except Exception as e:
        print(f"Error fetching comments: {e}")
    return comments


def clean_comments(raw_comments):
    """Filter out spam, links, and very short comments"""
    cleaned = []
    for c in raw_comments:
        text = c["text"].strip()
        if len(text.split()) >= 3 and not re.search(r"http[s]?://", text):
            cleaned.append(f"[{c['likes']} likes] {text}")
    return "\n".join(cleaned)


def analyze_with_ai(cleaned_text):
    """Generate structured feedback report using Gemini API"""
    prompt = f"""
    You are a professional YouTube Content Strategist.
    Below are comments and their like counts from a video:

    {cleaned_text}

    Analyze the feedback and provide a concise, actionable, and structured report for the creator:
    1. 🎯 Audience Demands (What topics/content do viewers want next?)
    2. ⚠️ Constructive Feedback / Criticisms (Issues regarding audio, pacing, clarity, editing, or depth)
    3. ⭐ Highlighted Segments (Specific topics, moments, or timestamps praised most)
    4. 🚀 3 Key Action Steps for the Next Video (To optimize engagement and retention)
    """

    response = ai_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )
    return response.text


if __name__ == "__main__":
    video_input = input("Enter YouTube Video URL or Video ID: ").strip()
    vid_id = extract_video_id(video_input)

    print("\n⏳ 1. Fetching comments from YouTube...")
    raw = fetch_comments(vid_id, max_count=80)

    if not raw:
        print(
            "❌ No comments found or invalid Video ID (comments might be disabled)."
        )
    else:
        print(
            f"✅ Found {len(raw)} comments. Cleaning and filtering spam..."
        )
        cleaned = clean_comments(raw)

        print("🤖 2. Analyzing with Gemini AI...\n")
        report = analyze_with_ai(cleaned)

        print("=" * 50)
        print("📊 YOUTUBE CREATOR FEEDBACK REPORT")
        print("=" * 50)
        print(report)