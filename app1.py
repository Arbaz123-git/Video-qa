import streamlit as st
from langchain_community.document_loaders import YoutubeLoader

#from langchain.document_loaders import YoutubeLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores.faiss import FAISS

#from langchain.vectorstores import FAISS
#from langchain.chains import LLMChain
from langchain_classic.chains import LLMChain

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import textwrap
from datetime import datetime

# Initialize embeddings
# With this:
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Streamlit app configuration
st.set_page_config(page_title="YouTube Video Q&A", layout="wide")
st.title("🎥 YouTube Video Q&A System")

# Initialize session state variables
if "db" not in st.session_state:
    st.session_state.db = None
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now()
if "video_urls" not in st.session_state:
    st.session_state.video_urls = []

def create_db_from_youtube_video_urls(video_urls):
    """Create FAISS database from multiple YouTube video URLs with optimized chunking"""
    if isinstance(video_urls, str):
        video_urls = [video_urls]
    
    all_docs = []
    
    for video_url in video_urls:
        try:
            st.info(f"Processing video: {video_url}")
            loader = YoutubeLoader.from_youtube_url(video_url)
            transcript = loader.load()
            
            for doc in transcript:
                doc.metadata['source_url'] = video_url
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300
            )
            docs = text_splitter.split_documents(transcript)
            all_docs.extend(docs)
            
        except Exception as e:
            st.error(f"Error processing {video_url}: {str(e)}")
            continue
    
    if not all_docs:
        raise ValueError("No documents were successfully processed from the provided URLs")
    
    db = FAISS.from_documents(all_docs, embeddings)
    return db

def enhanced_search_strategy(db, query, debug=False):
    """Enhanced search with multiple retrieval strategies"""
    all_relevant_docs = []
    
    direct_docs = db.similarity_search(query, k=8)
    all_relevant_docs.extend(direct_docs)
    
    intro_queries = [
        "my name is", "I am", "hello everyone", "welcome",
        "introduction", "presenter", "speaker"
    ]
    
    for intro_query in intro_queries:
        intro_docs = db.similarity_search(intro_query, k=3)
        all_relevant_docs.extend(intro_docs)
    
    person_queries = [
        "person", "name", "who", "presenter",
        "speaker", "instructor", "teacher"
    ]
    
    for person_query in person_queries:
        person_docs = db.similarity_search(person_query, k=2)
        all_relevant_docs.extend(person_docs)
    
    seen = set()
    unique_docs = []
    for doc in all_relevant_docs:
        doc_id = doc.page_content[:100]
        if doc_id not in seen:
            seen.add(doc_id)
            unique_docs.append(doc)
    
    final_docs = unique_docs[:12]
    
    if debug:
        st.write(f"🔍 Debug: Retrieved {len(final_docs)} unique chunks")
        for i, doc in enumerate(final_docs[:3]):
            st.write(f"Chunk {i+1} preview:")
            st.write(f"Content: {doc.page_content[:200]}...")
            st.write(f"Source: {doc.metadata.get('source_url', 'Unknown')}")
    
    return final_docs

def get_response_from_query(db, query, debug=False):
    """Enhanced query response with improved search strategy"""
    docs = enhanced_search_strategy(db, query, debug=debug)
    
    docs_page_content = " ".join([d.page_content for d in docs])
    
    max_content_length = 8000
    if len(docs_page_content) > max_content_length:
        docs_page_content = docs_page_content[:max_content_length] + "..."
        if debug:
            st.warning(f"Content truncated to {max_content_length} characters")
    
    chat = ChatGroq(
        api_key="GROQ_API_KEY", 
        model="llama-3.3-70b-versatile", 
        temperature=0
    )
    
    template = """You are a helpful assistant that can answer questions about youtube videos
        based on the video's transcript: {docs}
        
        IMPORTANT INSTRUCTIONS:
        - Carefully read through ALL the provided transcript content
        - Look for any mentions of names, introductions, or self-identification
        - Pay special attention to phrases like "my name is", "I am", "hello, I'm", etc.
        - If someone introduces themselves anywhere in the transcript, include that information
        - Only use the factual information from the transcript to answer the question
        - If you find relevant information, provide it even if it's not in the most obvious place
        - If you truly cannot find the information after carefully reviewing all content, say "I don't know"
        
        When referencing information, you can mention it comes from "one of the videos" or "the videos" since 
        the information may come from multiple video sources."""
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    
    return response, docs

def get_video_sources(docs):
    """Get unique video sources from the retrieved documents"""
    sources = set()
    for doc in docs:
        if 'source_url' in doc.metadata:
            sources.add(doc.metadata['source_url'])
    return list(sources)

# Streamlit UI
st.sidebar.header("Configuration")
video_urls_input = st.sidebar.text_area(
    "Enter YouTube Video URLs (one per line)",
    value="\n".join(st.session_state.video_urls),
    height=100
)
process_button = st.sidebar.button("Process Videos")

if process_button and video_urls_input:
    video_urls = [url.strip() for url in video_urls_input.split("\n") if url.strip()]
    st.session_state.video_urls = video_urls
    with st.spinner("Processing videos..."):
        try:
            st.session_state.db = create_db_from_youtube_video_urls(video_urls)
            st.success("✅ Database created successfully!")
        except Exception as e:
            st.error(f"❌ Error processing videos: {str(e)}")

st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode, 
                   key="debug_mode", help="Enable to see detailed search information")

# Main Q&A interface
st.header("💬 Ask a Question")
query = st.text_input("Enter your question about the videos:", key="query_input")

if st.button("Submit Question") and query and st.session_state.db:
    with st.spinner("Searching for answer..."):
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            response, docs = get_response_from_query(
                st.session_state.db, 
                query, 
                debug=st.session_state.debug_mode
            )
            st.session_state.question_history.append((timestamp, query, response))
            
            st.subheader("🤖 Answer")
            st.write(textwrap.fill(response, width=70))
            
            sources = get_video_sources(docs)
            if sources:
                st.subheader(f"📺 Sources ({len(sources)} video(s))")
                for source in sources[:3]:
                    st.write(f"• {source}")
                if len(sources) > 3:
                    st.write(f"... and {len(sources) - 3} more")
            
            if st.session_state.debug_mode:
                st.subheader("🔧 Debug Info")
                st.write(f"Retrieved {len(docs)} chunks")
                st.write(f"Total content length: {sum(len(d.page_content) for d in docs)} chars")
                
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")

# Display question history
if st.session_state.question_history:
    st.header("📝 Question History")
    for i, (timestamp, question, response) in enumerate(st.session_state.question_history, 1):
        with st.expander(f"{i}. [{timestamp}] {question}"):
            st.write(textwrap.fill(response, width=70))

# Display loaded video sources
if st.session_state.video_urls:
    st.header("📹 Loaded Video Sources")
    for i, source in enumerate(st.session_state.video_urls, 1):
        st.write(f"{i}. {source}")

# Session summary
st.header("📊 Session Summary")
session_duration = datetime.now() - st.session_state.session_start
st.write(f"Duration: {session_duration}")
st.write(f"Questions asked: {len(st.session_state.question_history)}")