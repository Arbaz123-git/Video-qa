import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores.faiss import FAISS
#from langchain.vectorstores import FAISS
from langchain_classic.chains import LLMChain
#from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import textwrap

# Set page configuration
st.set_page_config(
    page_title="YouTube Video Q&A Assistant",
    page_icon="🎥",
    layout="wide"
)

# Initialize embeddings (cache this to avoid reloading)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings()

def create_db_from_youtube_video_url(video_url):
    """Create vector database from YouTube video transcript"""
    try:
        loader = YoutubeLoader.from_youtube_url(video_url)
        transcript = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_documents(transcript)
        embeddings = load_embeddings()
        db = FAISS.from_documents(docs, embeddings)
        return db, None
    except Exception as e:
        return None, str(e)

def get_response_from_query(db, query, k=4):
    """Get response from the language model based on video content"""
    try:
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])

        # Initialize Groq chat (you might want to move API key to st.secrets)
        chat = ChatGroq(
        api_key="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",
        model="llama-3.3-70b-versatile", 
        temperature=0
        )

        # Create prompt templates
        template = """You are a helpful assistant that can answer questions about youtube videos
        based on the video's transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know"."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        human_template = "Answer the following question: {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        chain = LLMChain(llm=chat, prompt=chat_prompt)
        response = chain.run(question=query, docs=docs_page_content)
        response = response.replace("\n", "")
        
        return response, docs, None
    except Exception as e:
        return None, None, str(e)

# Main application
def main():
    st.title("🎥 YouTube Video Q&A Assistant")
    st.markdown("Ask questions about any YouTube video based on its transcript!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.markdown("""
        This app uses:
        - YouTube transcripts for content
        - FAISS for vector storage
        - Groq API for fast LLM responses
        - HuggingFace for embeddings
        """)
        
        st.info("💡 **Tip**: Use clear, specific questions for better answers!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Video Information")
        video_url = st.text_input(
            "YouTube Video URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the full YouTube URL here"
        )
        
        if video_url:
            st.video(video_url)
            
            if st.button("Load Video Transcript", type="primary"):
                with st.spinner("Loading transcript and processing video content..."):
                    db, error = create_db_from_youtube_video_url(video_url)
                    
                    if error:
                        st.error(f"Error loading video: {error}")
                    else:
                        st.session_state.db = db
                        st.success("✅ Video loaded successfully! You can now ask questions.")
    
    with col2:
        st.subheader("Ask Questions")
        
        if 'db' not in st.session_state:
            st.info("👈 Please load a YouTube video first to ask questions")
        else:
            query = st.text_input(
                "Your question:",
                placeholder="What is this video about? What are the key points?",
                help="Ask anything about the video content"
            )
            
            if st.button("Get Answer") and query:
                with st.spinner("Analyzing video content and generating answer..."):
                    response, docs, error = get_response_from_query(st.session_state.db, query)
                    
                    if error:
                        st.error(f"Error generating response: {error}")
                    else:
                        # Display response
                        st.markdown("### 🤖 Answer:")
                        st.markdown(f"**Question:** {query}")
                        st.markdown(f"**Answer:** {response}")
                        
                        # Show source information (optional)
                        with st.expander("🔍 View source details"):
                            st.write(f"Number of source chunks used: {len(docs)}")
                            for i, doc in enumerate(docs[:2]):  # Show first 2 sources
                                st.write(f"**Source {i+1}:**")
                                st.text(textwrap.fill(doc.page_content, width=80))
    
    # Example queries section
    if 'db' in st.session_state:
        st.markdown("---")
        st.subheader("💡 Example Questions You Can Ask")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("What is this video about?"):
                st.session_state.example_query = "What is this video about?"
                
        with col2:
            if st.button("What are the main topics?"):
                st.session_state.example_query = "What are the main topics covered in this video?"
                
        with col3:
            if st.button("Key takeaways?"):
                st.session_state.example_query = "What are the key takeaways from this video?"
        
        # Set the query from example if clicked
        if 'example_query' in st.session_state:
            st.query_params = {"query": st.session_state.example_query}
            # Force a rerun to process the query
            st.rerun()

# Handle query parameters for quick questions
def handle_query_params():
    params = st.query_params
    if "query" in params and 'db' in st.session_state:
        query = params["query"]
        # Process the query automatically
        response, docs, error = get_response_from_query(st.session_state.db, query)
        if not error:
            st.markdown("### 🤖 Answer:")
            st.markdown(f"**Question:** {query}")
            st.markdown(f"**Answer:** {response}")

if __name__ == "__main__":
    main()
    handle_query_params()