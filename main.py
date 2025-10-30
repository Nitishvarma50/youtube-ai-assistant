import os
from dotenv import load_dotenv
from utils import extract_video_id, fetch_transcript
from rag_system import RAGSystem


def main():
    """Main application function."""
    # Load environment variables
    load_dotenv()
    
    # Verify API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    # Get YouTube URL from user
    url = input("Enter YouTube URL: ")
    
    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        print("❌ Invalid YouTube URL")
        return
    
    print(f"🎥 Video ID: {video_id}")
    
    # Fetch transcript
    try:
        transcript_text = fetch_transcript(video_id)
        print(f"📝 Transcript preview: {transcript_text[:500]}...\n")
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return
    
    # Initialize RAG system
    rag = RAGSystem(chunk_size=1000, chunk_overlap=200, k=4, temperature=0.2)
    
    # Index the transcript
    rag.index_transcript(transcript_text)
    
    # Interactive Q&A loop
    print("\n🤖 RAG System ready! Ask questions about the video (type 'quit' to exit)\n")
    
    while True:
        question = input("Your question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not question.strip():
            continue
        
        try:
            answer = rag.query(question)
            print(f"\n💡 Answer: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    main()
