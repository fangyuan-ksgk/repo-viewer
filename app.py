from tools.repo import *
from tools.agent import RepoAgent, clone_repo
import gradio as gr


def analyze_repo(repo_url):
    temp_repo = decide_temp_repo(repo_url)
    clone_repo(repo_url, temp_repo)
    agent = RepoAgent(temp_repo)
    
    # Visualize file dependency
    file_dep_img = agent.visualize_file(cap_node_number=25)
    
    # Visualize module dependency for the first Python file
    python_files = [f for f in agent.files if f.endswith('.py')]
    if python_files:
        module_dep_img = agent.visualize_module(python_files[0], cap_node_number=10, depth=3)
    else:
        module_dep_img = None
    
    # Generate podcast
    podcast = agent.generate_podcast()
    
    return file_dep_img, module_dep_img, podcast

# Create Gradio interface
iface = gr.Interface(
    fn=analyze_repo,
    inputs=gr.Textbox(label="GitHub Repository URL"),
    outputs=[
        gr.Image(label="File Dependency Graph"),
        gr.Image(label="Module Dependency Graph"),
        gr.Audio(label="Generated Podcast")
    ],
    title="GitHub Repository Analyzer",
    description="Enter a GitHub repository URL to analyze its structure and generate a podcast."
)

# Launch the interface
iface.launch()