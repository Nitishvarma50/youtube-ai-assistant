import streamlit as st
import os
from dotenv import load_dotenv
from utils import extract_video_id, fetch_transcript
from rag_system import RAGSystem

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF0000;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTextInput > label {
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎥 YouTube RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about any YouTube video using AI</p>', unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for video input and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key check
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found in .env file")
        st.stop()
    else:
        st.success("✅ API Key loaded")
    
    st.divider()
    
    # YouTube URL input
    st.subheader("📹 Video Input")
    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=..."
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        k_retrieval = st.slider("Number of Chunks to Retrieve", 2, 10, 4, 1)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    
    # Process video button
    if st.button("🚀 Process Video", type="primary", use_container_width=True):
        if not youtube_url:
            st.error("Please enter a YouTube URL")
        else:
            with st.spinner("Processing video..."):
                try:
                    # Extract video ID
                    video_id = extract_video_id(youtube_url)
                    if not video_id:
                        st.error("❌ Invalid YouTube URL")
                        st.stop()
                    
                    st.session_state.video_id = video_id
                    st.info(f"🎥 Video ID: {video_id}")
                    
                    # Fetch transcript
                    transcript_text = fetch_transcript(video_id)
                    st.success("✅ Transcript fetched successfully!")
                    
                    # Initialize RAG system
                    rag = RAGSystem(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        k=k_retrieval,
                        temperature=temperature
                    )
                    
                    # Index transcript
                    rag.index_transcript(transcript_text)
                    st.session_state.rag_system = rag
                    st.session_state.indexed = True
                    st.session_state.chat_history = []  # Reset chat history
                    
                    st.success("🎉 Video indexed! You can now ask questions.")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.indexed = False
    
    # Clear conversation button
    if st.session_state.indexed:
        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Main content area
if not st.session_state.indexed:
    # Welcome message
    st.info("👈 Enter a YouTube URL in the sidebar and click 'Process Video' to get started!")
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Smart Retrieval")
        st.write("Uses vector embeddings to find the most relevant parts of the video")
    
    with col2:
        st.markdown("### 💬 Natural Q&A")
        st.write("Ask questions in plain English and get accurate answers")
    
    with col3:
        st.markdown("### 🚀 Fast Processing")
        st.write("Powered by OpenAI GPT and FAISS vector search")
    
    # Instructions
    st.divider()
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter a YouTube URL in the sidebar
    2. Click "Process Video" to index the transcript
    3. Ask questions about the video content
    4. Get AI-powered answers based on the transcript
    """)

else:
    # Chat interface
    st.markdown(f"### 💬 Chat about Video: `{st.session_state.video_id}`")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
    
    # Question input
    question = st.chat_input("Ask a question about the video...")
    
    if question:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(question)
        
        # Get answer from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_system.query(question)
                    st.write(answer)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Built with ❤️ using Streamlit, LangChain, and OpenAI | 
    <a href='https://github.com' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)
