import rag_agent
import asyncio
import streamlit as st
import atexit
from utils import download_uploaded_file, get_articles, clear_docs, load_css


def intro():
    #NOTE new stuff below here:
     # Custom header with cyberpunk vibe
    st.markdown("""
        <div class="custom-header fade-in">
            <h1>⚡ STUDY MUNCH AI ⚡</h1>
            <p>RAG Agent Learning Interface</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Welcome content in glowing cards
    st.markdown("""
        <div class="info-card fade-in">
            <h2>🚀 SYSTEM ACTIVE</h2>
            <p>Welcome to the next generation of AI-powered learning. Study Munch AI combines advanced RAG technology with intelligent document analysis to create your ultimate study companion.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="info-card fade-in">
                <h3>⚙️ CORE FEATURES</h3>
                <p>
                    <strong>🔍 Deeper Context</strong><br/>
                    Automatically discovers and indexes relevant articles related to your topic<br/><br/>
                    </strong>
                    <strong>📊 Document Analysis</strong><br/>
                    Deep semantic understanding of your materials<br/><br/>
                    </strong>
                    <strong>🧠 Adaptive Memory</strong><br/>
                    Context-aware conversation tracking
                    </strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-card fade-in">
                <h3>🎯 CAPABILITIES</h3>
                <p>
                    <strong>💬 Intelligent Dialogue</strong><br/>
                    Not just Q&A - real conversations<br/><br/>
                    </strong>
                    <strong>🎨 Customizable</strong><br/>
                    Adjust response style and depth<br/><br/>
                    </strong>
                    <strong>⚡ Real-Time Learning</strong><br/>
                    Fetch fresh information on demand
                    </strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card fade-in pulse-glow">
            <h3>🎓 HOW TO ACTIVATE</h3>
            <p>
                <strong>STEP 1:</strong> Upload your study materials (PDFs, TXT, CSV)<br/>
                <strong>STEP 2:</strong> Define your learning objective<br/>
                <strong>STEP 3:</strong> Configure system parameters (optional)<br/>
                <strong>STEP 4:</strong> Begin study session
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("👈 Navigate to **CHAT ROOM** via sidebar to initialize your learning session")


@st.fragment()
def chat_room():
    # Initialize session variables - preface stage
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
        st.session_state.uploaded_files = []
        st.session_state.topic = ""
    
    # Get user to upload any necessary context files and give it topic to gather information on
    if not st.session_state.setup_complete:
        st.markdown("""
            <div class="custom-header fade-in">
                <h1>⚡ SYSTEM CONFIGURATION ⚡</h1>
                <p>Initialize your learning parameters</p>
            </div>
        """, unsafe_allow_html=True)

        with st.form("setup_form", clear_on_submit=False):
            st.markdown("### 📂 UPLOAD CONTEXT FILES")
            st.markdown("*Enhance the knowledge base with your materials*")
            
            uploaded_files = st.file_uploader(
                "Drop files here", 
                type=["txt","pdf", "csv"],
                accept_multiple_files=True,
                help="Maximum 3 files for optimal performance")
            max_file_size_mb = 1 * 1024 *1024 # 1 MB
            if uploaded_files:
                # --- 1. Validate File Count ---
                if len(uploaded_files) > 3:
                    st.error(f"⚠️ LIMIT EXCEEDED: Maximum 3 files allowed")
                    
                else:
                    # --- 2. Validate File Size ---
                    all_valid = True
                    
                    for file in uploaded_files:
                        if file.size > max_file_size_mb:
                            st.error(f"❌ FILE TOO LARGE: File '{file.name}' is {file.size / (1024 * 1024):.2f} MB. Maximum allowed is 1 MB.")
                            all_valid = False
                        
                    # --- 3. Process Valid Files ---
                    if all_valid:
                        # Filter out files that were validated successfully
                        valid_files = [file for file in uploaded_files if file.size <= max_file_size_mb]
                        st.success(f"✅ {len(valid_files)} FILE(S) LOADED SUCCESSFULLY")
          
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            
            st.markdown("### 🎯 DEFINE LEARNING OBJECTIVE")
            topic = st.text_input(
                "Subject Area", 
                placeholder="Enter your study topic...",
                help="This optimizes content retrieval and conversation focus")
            
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            
            # Advanced settings
            with st.expander("⚙️ ADVANCED CONFIGURATION"):
                st.markdown("**MODEL SELECTION**")
                model_choice = st.selectbox(
                    "Agent Model",
                    options=["gpt-3.5-turbo", "gpt-4.1"],
                    index=0,
                    help="GPT-4.1: Superior reasoning | GPT-3.5: Faster, cost-efficient"
                )
                
                st.markdown("**RESPONSE PARAMETERS**")
                
                temperature = st.slider(
                    "Creativity Index",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Lower = focused | Higher = creative"
                )
                
                max_response_tokens = st.slider(
                    "Response Depth (tokens)",
                    min_value=256,
                    max_value=2000,
                    value=1000,
                    step=256,
                    help="~750 tokens ≈ 500 words"
                )
                
                st.markdown("**SESSION LIMITS**")
                token_limit = st.number_input(
                    "Total token budget", 
                    min_value=1000, 
                    max_value=50000, 
                    value=10000,
                    step=1000,
                    help="Maximum tokens for entire conversation"
                )
            
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            
            submitted = st.form_submit_button("🚀 INITIALIZE SYSTEM", use_container_width=True)
            
            if submitted:
                # Needs a topic
                if not topic.strip(): 
                    st.error("⚠️ ERROR: Learning objective required")
                else: 
                    with st.spinner("⚡ INITIALIZING AGENT..."):
                        # When files are uploaded
                        if uploaded_files:
                            for f in uploaded_files:
                                download_uploaded_file(file=f)
                        st.session_state.topic = topic
                        get_articles(st.session_state.topic)
                        st.session_state.setup_complete = True
                        st.session_state.agent = rag_agent.agent(
                            verbose=True,
                            llm_model=model_choice,
                            topic=st.session_state.topic,
                            token_limit=token_limit,
                            temperature=temperature,
                            max_response_tokens=max_response_tokens
                        )
                    st.success("✅ SYSTEM ONLINE")
                    st.rerun(scope="fragment")

    # The actual chat room
    else:
        st.markdown(f"""
            <div class="custom-header fade-in">
                <h1>⚡ MUNCHING ON: {st.session_state.topic.upper()} ⚡</h1>
                <p>Agent ready for queries</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Token usage display with glowing metrics
        # if hasattr(st.session_state.agent, 'get_token_usage_info'):
        #     usage_info = st.session_state.agent.get_token_usage_info()
            
            # col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            # with col1:
            #     st.progress(
            #         usage_info['percent_of_limit_used'] / 100,
            #         text=f"⚡ TOKEN USAGE: {usage_info['total_tokens']:,} / {usage_info['token_limit']:,}"
            #     )
            
            # with col2:
            #     st.metric("💾 REMAINING", f"{usage_info['remaining_tokens']:,}")
            
            # with col3:
            #     percentage = f"{usage_info['percent_of_limit_used']:.1f}%"
            #     st.metric("📊 USED", percentage)
            
            # with col4:
            #     status = "🟢 OPTIMAL" if usage_info['percent_of_limit_used'] < 80 else "🟡 HIGH"
            #     st.metric("STATUS", status)
        
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        
        # Control panel
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            reset_conversation = st.button("🔄 NEW SESSION", use_container_width=True)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the chat history
        for entry in st.session_state.messages:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
        
        # User input entry
        user_input = st.chat_input("⚡ Enter your query...")
        
        if user_input:
            # Validate
            if len(user_input.strip()) == 0 or len(user_input) > 500:
                st.error("⚠️ INPUT ERROR: Message must be 1-500 characters")
                st.rerun(scope="fragment")
            
            # Display user input
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
          
            # RAG agent's response
            with st.spinner("🧠 MUNCHING QUERY..."):
                r_response = asyncio.run(st.session_state.agent(user_input))
            st.session_state.messages.append({"role": "assistant", "content": r_response})
            # Display agent's response
            with st.chat_message("assistant"):
                st.markdown(r_response)
                st.rerun(scope="fragment")
            

        # Returns back to prefacing stage
        if reset_conversation:
            st.session_state.setup_complete = False
            clear_docs()
            st.session_state.messages = []
            st.rerun(scope="fragment")

if __name__ == "__main__":
    load_css()
    atexit.register(clear_docs)# clear docs when done, crtl + C in terminal
    st.set_page_config(
        page_title="Study Munch AI - Agent Interface",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar with glowing header
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 1.5rem; background: rgba(0, 255, 255, 0.05); border-radius: 15px; border: 2px solid rgba(0, 255, 255, 0.3); box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);'>
            <h1 style='color: #00ffff; text-shadow: 0 0 15px rgba(0, 255, 255, 0.8); margin: 0;'>⚡ STUDY MUNCH</h1>
            
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    
    page_names_to_funcs = {
        "🏠 Welcome Page": intro,
        "💬 Chat Room": chat_room
    }

    selected_page = st.sidebar.selectbox(
        "NAVIGATION",
        options=page_names_to_funcs.keys(),
        key="current_page"
    )
    
    st.sidebar.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div style='padding: 1rem; background: rgba(0, 255, 255, 0.05); border-radius: 12px; border: 1px solid rgba(0, 255, 255, 0.3);'>
            <p style='font-size: 0.85rem; margin: 0; color: rgba(0, 255, 255, 0.9);'>
                <strong>⚡ SYSTEM TIP:</strong><br/>
                Upload documents to enhance the agents knowledge base when you start your session!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    
    st.sidebar.markdown("""
        <div style='text-align: center; padding: 0.5rem; color: rgba(0, 255, 255, 0.5); font-size: 0.75rem;'>
            SYSTEM STATUS: <span style='color: #00ff66;'>●</span> ONLINE
        </div>
    """, unsafe_allow_html=True)
    
    page_names_to_funcs[selected_page]()