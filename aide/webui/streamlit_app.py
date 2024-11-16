import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import tempfile
import shutil
import os
import json
from omegaconf import OmegaConf
from rich.console import Console
import sys
import textwrap
import numpy as np
from igraph import Graph

from aide.webui.tree_viz import create_tree_visualization
from aide import Experiment
from aide.utils.tree_export import generate_html, cfg_to_tree_struct, generate as tree_export_generate
from aide.utils.config import load_task_desc, prep_agent_workspace, save_run, prep_cfg
from aide.journal import Journal
from aide.agent import Agent
from aide.interpreter import Interpreter

console = Console(file=sys.stderr)

def load_example_files():
    """Load the house prices example files into memory"""
    example_dir = Path(__file__).parent / "example_tasks" / "house_prices"
    example_files = []
    
    for file_path in example_dir.glob("*"):
        if file_path.suffix.lower() in ['.csv', '.txt', '.json', '.md']:
            # Create a NamedTemporaryFile object with the file content
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_path.suffix) as tmp_file:
                tmp_file.write(file_path.read_bytes())
                example_files.append({
                    'name': file_path.name,
                    'path': tmp_file.name
                })
    
    return example_files

def run_aide(files, goal_text, eval_text, num_steps):
    try:
        # Set API keys from session state
        if st.session_state.get('openai_key'):
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
        if st.session_state.get('anthropic_key'):
            os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key

        # Create input directory
        project_root = Path(__file__).parent.parent.parent
        input_dir = project_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
            
        # Handle uploaded files
        if files:
            for file in files:
                if isinstance(file, dict):  # Example files
                    shutil.copy2(file['path'], input_dir / file['name'])
                else:  # Uploaded files
                    with open(input_dir / file.name, 'wb') as f:
                        f.write(file.getbuffer())
        else:
            st.error("Please upload data files")
            return None

        # Initialize experiment
        experiment = Experiment(
            data_dir=str(input_dir),
            goal=goal_text,
            eval=eval_text,
            steps=num_steps
        )
        
        # Run experiment with progress bar
        progress_bar = st.progress(0)
        for step in range(num_steps):
            progress_bar.progress((step + 1) / num_steps)
            experiment.step()

        return {
            "solution": (experiment.cfg.workspace_dir / "best_solution.py").read_text(),
            "config": OmegaConf.to_yaml(experiment.cfg),
            "journal": json.dumps(experiment.journal.to_dict(), indent=2),
            "tree_path": str(experiment.cfg.log_dir / "tree_plot.html")
        }

    except Exception as e:
        console.print_exception()
        st.error(f"Error occurred: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AIDE: AI Development Environment", layout="wide")

    # Title and description
    st.title("AIDE: AI Development Environment")
    st.markdown("An LLM agent that generates solutions for machine learning tasks from natural language descriptions.")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Keys
        st.session_state.openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", "")
        )
        st.session_state.anthropic_key = st.text_input(
            "Anthropic API Key (Optional)",
            type="password",
            value=os.getenv("ANTHROPIC_API_KEY", "")
        )

        # Load example button
        if st.button("Load Example"):
            example_data = load_example_files()
            st.session_state.update(example_data)

    # Main content area
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")
        
        # File uploader
        if st.session_state.example_files:
            st.info("Example files loaded! Click 'Run AIDE' to proceed.")
            # Display the names of loaded example files
            st.write("Loaded files:")
            for file in st.session_state.example_files:
                st.write(f"- {file['name']}")
            uploaded_files = st.session_state.example_files
        else:
            uploaded_files = st.file_uploader(
                "Upload Data Files",
                accept_multiple_files=True,
                type=["csv", "txt", "json", "md"]
            )
        
        goal_text = st.text_area(
            "Goal",
            value=st.session_state.get("goal", ""),
            placeholder="Example: Predict house prices"
        )
        
        eval_text = st.text_area(
            "Evaluation Criteria",
            value=st.session_state.get("eval", ""),
            placeholder="Example: Use RMSE metric"
        )
        
        num_steps = st.slider(
            "Number of Steps",
            min_value=1,
            max_value=20,
            value=st.session_state.get("steps", 10)
        )

        if st.button("Run AIDE", type="primary"):
            results = run_aide(uploaded_files, goal_text, eval_text, num_steps)
            if results:
                st.session_state.results = results

    with col2:
        st.header("Results")
        tabs = st.tabs(["Best Solution", "Config", "Journal", "Tree Visualization"])
        
        if "results" in st.session_state:
            results = st.session_state.results
            
            with tabs[0]:
                st.code(results["solution"], language="python")
            
            with tabs[1]:
                st.code(results["config"], language="yaml")
            
            with tabs[2]:
                st.code(results["journal"], language="json")
            
            with tabs[3]:
                if os.path.exists(results["tree_path"]):
                    with open(results["tree_path"], 'r', encoding='utf-8') as f:
                        components.html(f.read(), height=600, scrolling=True)

if __name__ == "__main__":
    main() 