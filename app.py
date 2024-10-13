import streamlit as st
from tools.repo import *
from tools.agent import RAgent, clone_repo
import os

st.set_page_config(page_title="RepoViewer", layout="wide")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None

# Sidebar for repository input and analysis functions
st.sidebar.title("RepoViewer")
repo_url = st.sidebar.text_input("GitHub Repository URL", "https://github.com/xjdr-alt/entropix.git")

if st.sidebar.button("Analyze Repository"):
    with st.spinner("Cloning and analyzing repository..."):
        temp_repo = decide_temp_repo(repo_url)
        clone_repo(repo_url, temp_repo)
        st.session_state.agent = RAgent(temp_repo)
    st.sidebar.success("Repository analyzed successfully!")

# Visualization functions in sidebar
if st.session_state.agent:
    st.sidebar.header("Visualizations")
    
    if st.sidebar.button("File Dependency Visualization"):
        with st.spinner("Generating file dependency visualization..."):
            static_viz = st.session_state.agent.visualize_file()
            st.image(static_viz, caption="Static File Dependency Visualization", use_column_width=True)

    st.sidebar.subheader("Module Dependency Visualization")
    file_list = st.session_state.agent.get_file_list()
    selected_file = st.sidebar.selectbox("Select a file to visualize", file_list)
    if selected_file and st.sidebar.button("Generate Module Visualization"):
        with st.spinner("Generating module dependency visualization..."):
            module_viz = st.session_state.agent.visualize_module(selected_file, cap_node_number=10, depth=3)
            st.image(module_viz, caption=f"Module Dependency for {selected_file}", use_column_width=True)

    # Podcast Generation
    if st.sidebar.button("Generate Repository Podcast"):
        with st.spinner("Generating podcast..."):
            podcast_path = st.session_state.agent.generate_podcast()
        st.audio(podcast_path)


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at the bottom
if prompt := st.chat_input("Chat with RepoAgent about the repository..."):
    if not st.session_state.agent:
        st.error("Please analyze a repository first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.respond(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Cleanup button (moved to the bottom of the sidebar)
if st.sidebar.button("Cleanup Temporary Files"):
    with st.spinner("Cleaning up..."):
        # Add cleanup logic here (e.g., removing temp directories)
        st.success("Cleanup completed!")