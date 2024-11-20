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
from dotenv import load_dotenv

from aide import Experiment
from aide.utils.tree_export import generate_html, cfg_to_tree_struct, generate as tree_export_generate
from aide.utils.config import load_task_desc, prep_agent_workspace, save_run, prep_cfg, load_cfg
from aide.journal import Journal
from aide.agent import Agent
from aide.interpreter import Interpreter

console = Console(file=sys.stderr)

def load_env_variables():
    """Load environment variables from .env file"""
    # Load from .env file if it exists
    load_dotenv()
    
    # Get API keys from environment with fallback to empty string
    return {
        'openai_key': os.getenv("OPENAI_API_KEY", ""),
        'anthropic_key': os.getenv("ANTHROPIC_API_KEY", "")
    }

def load_example_files():
    """Load the house prices example files into memory"""
    # Get the package directory where example tasks are stored
    package_root = Path(__file__).parent.parent
    example_dir = package_root / "example_tasks" / "house_prices"
    
    if not example_dir.exists():
        st.error(f"Example directory not found at: {example_dir}")
        return []
        
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
    
    if not example_files:
        st.warning("No example files found in the example directory")
    
    # Set example goal and eval criteria from README
    st.session_state["goal"] = "Predict the sales price for each house"
    st.session_state["eval"] = "Use the RMSE metric between the logarithm of the predicted and observed values."
    
    return example_files

def run_aide(files, goal_text, eval_text, num_steps):
    try:
        # Set running state
        st.session_state.is_running = True
        st.session_state.current_step = 0  # Initialize step counter
        
        # Set API keys from session state
        if st.session_state.get('openai_key'):
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_key
        if st.session_state.get('anthropic_key'):
            os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_key

        # Create input directory in the project root
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
            eval=eval_text
        )
        
        # Store config in session state immediately
        st.session_state.current_config = OmegaConf.to_yaml(experiment.cfg)

        # Run experiment with progress updates
        for step in range(num_steps):
            st.session_state.current_step = step + 1  # Update step counter
            progress = (step + 1) / num_steps
            st.session_state.progress = progress
            experiment.run(steps=1)

        # Clear running state
        st.session_state.is_running = False
        
        return {
            "solution": (experiment.cfg.log_dir / "best_solution.py").read_text() if (experiment.cfg.log_dir / "best_solution.py").exists() else "No solution found",
            "config": OmegaConf.to_yaml(experiment.cfg),
            "journal": json.dumps(experiment.journal.to_dict(), indent=2),
            "tree_path": str(experiment.cfg.log_dir / "tree_plot.html")
        }

    except Exception as e:
        st.session_state.is_running = False
        console.print_exception()
        st.error(f"Error occurred: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="AIDE: AI Development Environment", layout="wide")
    
    # Load environment variables at startup
    env_vars = load_env_variables()
    
    # Initialize results
    results = {}
    
    # Title and description
    st.title("AIDE: AI Development Environment")
    st.markdown("An LLM agent that generates solutions for machine learning tasks from natural language descriptions.")

    # Initialize session state for example files if it doesn't exist
    if 'example_files' not in st.session_state:
        st.session_state.example_files = []

    # Main content area - top row for input and config
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Input")
        
        # Create two columns for the buttons
        button_col1, button_col2 = st.columns(2)
        
        # Example files button
        with button_col1:
            if st.button("Load Example Experiment", use_container_width=True):
                st.session_state.example_files = load_example_files()
        
        # API Keys button
        with button_col2:
            with st.expander("Load API Keys", expanded=False):
                openai_key = st.text_input(
                    "OpenAI API Key",
                    value=env_vars['openai_key'],
                    type="password"
                )
                anthropic_key = st.text_input(
                    "Anthropic API Key",
                    value=env_vars['anthropic_key'],
                    type="password"
                )
                if st.button("Save API Keys", use_container_width=True):
                    st.session_state.openai_key = openai_key
                    st.session_state.anthropic_key = anthropic_key
                    st.success("API keys saved!")
        
        # File uploader
        if st.session_state.example_files:
            st.info("Example files loaded! Click 'Run AIDE' to proceed.")
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

    with col2:
        st.header("Configuration")
        
        # Create placeholders for dynamic content
        status_placeholder = st.empty()
        step_placeholder = st.empty()
        config_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Show running status and config
        if st.session_state.get("is_running", False):
            # Show status
            status_placeholder.markdown("### 🔄 AIDE is working...")
            
            # Show step counter
            current_step = st.session_state.get("current_step", 0)
            total_steps = st.session_state.get("total_steps", num_steps)
            step_placeholder.markdown(f"**Step {current_step}/{total_steps}**")
            
            # Show configuration
            if "current_config" in st.session_state:
                config_placeholder.code(st.session_state.current_config, language="yaml")
            
            # Show progress bar
            progress = st.session_state.get("progress", 0)
            progress_placeholder.progress(progress)

    # Results section below
    st.header("Results")
    tabs = st.tabs(["Best Solution", "Config", "Journal", "Tree Visualization"])
    
    with tabs[0]:
        if results and "solution" in results:
            st.code(results["solution"], language="python")
    
    with tabs[1]:
        if results and "config" in results:
            st.code(results["config"], language="yaml")
    
    with tabs[2]:
        if results and "journal" in results:
            st.code(results["journal"], language="json")
    
    with tabs[3]:
        if results and "tree_path" in results and os.path.exists(results["tree_path"]):
            with open(results["tree_path"], 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600, scrolling=True)

if __name__ == "__main__":
    main() 