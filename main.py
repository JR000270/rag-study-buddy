import rag_agent
import asyncio
import streamlit as st
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import atexit
from utils import download_uploaded_file, get_articles, clear_docs


def intro():
    st.title("Welcome to Jadens RAG Agent")
    st.write("""Use sidebar to navigate to the chat demo!
                From there you will be asked to drop any files you want the agent to have as context (optional)
                and to give it a topic (needed)
                Once you do that you can talk to it about what you need from there!""")

@st.fragment()
def chat_room():
# Initialize session variables
    #preface stage:
    if "setup_complete" not in st.session_state:
        st.session_state.setup_complete = False
        st.session_state.uploaded_files = []
        st.session_state.topic = ""
    
    #get user to upload any necessary context files and give it topic so gather information on
    if not st.session_state.setup_complete:
        st.subheader("Welcome User!")
        st.write("To get started just upload your necessary files for context and let me know what your topic is!\n")

        with st.form("setup_form", clear_on_submit=False):
            uploaded_files = st.file_uploader(
                "Upload documents", 
                type=["txt","pdf", "csv"],
                accept_multiple_files=True,
                help = "File dropbox")
            
            topic = st.text_input("What's the topic you need help with today?", help= "Topic input")
            
            submitted = st.form_submit_button("Start Chat")
            if submitted:
                #needs a topic
                if not topic.strip(): 
                    st.error("Please give me a topic")
                else: 
                    #when files are uploaded
                    if uploaded_files:
                        for f in uploaded_files:
                            download_uploaded_file(file = f)
                    st.session_state.topic = topic
                    get_articles(st.session_state.topic)
                    st.session_state.setup_complete = True
                    st.session_state.agent = rag_agent.agent(verbose=True, topic = st.session_state.topic)
                    st.rerun(scope="fragment")

    #the actual chat room
    else:
        st.subheader(f"Chatting about {st.session_state.topic}")
        reset_conversation = st.button(label="New Conversation")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the chat history
        for entry in st.session_state.messages:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
        

        #user input entry
        user_input = st.chat_input("Type message")
        if user_input:
            #display user input
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
          
            #rag agents response
            r_response = asyncio.run(st.session_state.agent(user_input))

            #display agents response
            st.session_state.messages.append({"role": "assistant", "content": r_response})
            with st.chat_message("assistant"):
                st.markdown(r_response)
                st.rerun(scope="fragment")

        #returns back to prefacing stage
        if reset_conversation:
            st.session_state.setup_complete = False
            clear_docs()
            st.session_state.messages = []
            st.rerun(scope="fragment")



if __name__ == "__main__":
    atexit.register(clear_docs)# clear docs when done, crtl + C in terminal
    st.set_page_config(page_title="Study Munch AI")
    st.title("Study Munch")

    st.sidebar.title("Menu")
    page_names_to_funcs = {
        "Welcome Page": intro,
        "Chat Room" : chat_room
    }

    selected_page = st.sidebar.selectbox("Choose a page", options=page_names_to_funcs.keys(), key= "current_page")
    page_names_to_funcs[selected_page]()