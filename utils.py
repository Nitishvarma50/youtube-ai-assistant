import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound


def extract_video_id(url: str) -> str:
    """
    Extracts the YouTube video ID from a given URL.
    Works for formats like:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
    """
    pattern = r"(?:v=|/)([0-9A-Za-z_-]{11})(?:[&?]|$)"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def fetch_transcript(video_id: str) -> str:
    """
    Fetches the transcript for a YouTube video.
    Tries English first, then falls back to auto-generated or other languages.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Full transcript text as a string
    """
    ytt_api = YouTubeTranscriptApi()
    
    try:
        # Try English first
        fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
    except NoTranscriptFound:
        print("⚠️ No English transcript found, trying auto or other languages...")
        try:
            # Try auto-generated or other language transcripts
            fetched_transcript = ytt_api.fetch(video_id, languages=["auto", "hi", "en"])
        except Exception as e:
            raise RuntimeError(f"❌ Transcript not available for this video: {e}")
    
    # Join transcript text
    full_text = " ".join([snippet.text for snippet in fetched_transcript.snippets])
    
    print("✅ Transcript successfully extracted!\n")
    return full_text
