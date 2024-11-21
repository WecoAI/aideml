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
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # Stream handler to write to stderr (will show in Streamlit)
        logging.StreamHandler(sys.stderr)
    ]
)

# Get the aide logger
logger = logging.getLogger("aide")
logger.setLevel(logging.INFO)

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
            # Store description file content separately
            if file_path.name == "data_description.txt":
                desc_content = file_path.read_text()
            
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

def run_aide(files, goal_text, eval_text, num_steps, results_col):
    try:
        # Create placeholders in the results column
        with results_col:
            status_placeholder = st.empty()
            step_placeholder = st.empty()
            config_title_placeholder = st.empty()
            config_placeholder = st.empty()
            progress_placeholder = st.empty()
        # Initialize session state
        st.session_state.is_running = True
        st.session_state.current_step = 0
        st.session_state.total_steps = num_steps
        st.session_state.progress = 0
        
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
            eval=eval_text
        )
        
        # Update status and config immediately in results column
        step_placeholder.markdown(f"### 🔥 Running Step {st.session_state.current_step}/{num_steps}")
        config_title_placeholder.markdown("### 📋 Configuration")
        config_placeholder.code(OmegaConf.to_yaml(experiment.cfg), language="yaml")
        progress_placeholder.progress(0)

        # Run experiment with progress updates
        for step in range(num_steps):
            st.session_state.current_step = step + 1
            progress = (step + 1) / num_steps
            
            # Update UI in results column
            with results_col:
                step_placeholder.markdown(f"### 🔥 Running Step {st.session_state.current_step}/{num_steps}")
                progress_placeholder.progress(progress)
            
            experiment.run(steps=1)

        # Clear running state and status messages
        st.session_state.is_running = False
        status_placeholder.empty()      # Clear the "AIDE is working..." message
        step_placeholder.empty()        # Clear step counter
        config_placeholder.empty()      # Clear config
        config_title_placeholder.empty()
        progress_placeholder.empty()    # Clear progress bar
        
        return {
            "solution": (experiment.cfg.log_dir / "best_solution.py").read_text() if (experiment.cfg.log_dir / "best_solution.py").exists() else "No solution found",
            "config": OmegaConf.to_yaml(experiment.cfg),
            "journal": json.dumps([{
                "step": node.step,
                "code": str(node.code),
                "metric": str(node.metric.value) if node.metric else None,
                "is_buggy": node.is_buggy
            } for node in experiment.journal.nodes], indent=2, default=str),
            "tree_path": str(experiment.cfg.log_dir / "tree_plot.html")
        }

    except Exception as e:
        st.session_state.is_running = False
        console.print_exception()
        st.error(f"Error occurred: {str(e)}")
        return None

def load_css():
    """Load custom CSS from streamlit_app_style.css"""
    css_file = Path(__file__).parent / "streamlit_app_style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found at: {css_file}")

def main():
    st.set_page_config(page_title="AIDE: the Machine Learning Engineer Agent", layout="wide")
    
    # Load custom CSS
    load_css()
    
    # Add a settings menu in the sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        env_vars = load_env_variables()
        
        # API Keys in sidebar
        st.markdown("<p style='text-align: center;'>OpenAI API Key</p>", unsafe_allow_html=True)
        openai_key = st.text_input(
            "OpenAI API Key",
            value=env_vars['openai_key'],
            type="password",
            label_visibility="collapsed"
        )
        
        st.markdown("<p style='text-align: center;'>Anthropic API Key</p>", unsafe_allow_html=True)
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=env_vars['anthropic_key'],
            type="password",
            label_visibility="collapsed"
        )
        
        if st.button("Save API Keys", use_container_width=True):
            st.session_state.openai_key = openai_key
            st.session_state.anthropic_key = anthropic_key
            st.success("API keys saved!")
    
    # Title and description
    st.title("AIDE: the Machine Learning Engineer Agent")
    st.markdown("An LLM agent that generates solutions for machine learning tasks just from natural language descriptions of the task.")    
    # Create columns for input and results (1:2 ratio)
    input_col, results_col = st.columns([1, 2])
    
    with input_col:
        st.header("Input")
        
        # Load example button
        if st.button("Load Example Experiment", use_container_width=True):
            st.session_state.example_files = load_example_files()
        
        # File uploader and other inputs
        if st.session_state.get('example_files'):
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

        # Run button and execution
        if st.button("Run AIDE", type="primary", use_container_width=True):
            with st.spinner("AIDE is running..."):
                results = run_aide(uploaded_files, goal_text, eval_text, num_steps, results_col)
                st.session_state.results = results

    # Results column
    with results_col:
        st.header("Results")
        if st.session_state.get('results'):
            results = st.session_state.results
            tabs = st.tabs(["Best Solution", "Config", "Journal", "Tree Visualization"])
            
            with tabs[0]:
                if "solution" in results:
                    st.code(results["solution"], language="python")
            
            with tabs[1]:
                if "config" in results:
                    st.code(results["config"], language="yaml")
            
            with tabs[2]:
                if "journal" in results:
                    st.code(results["journal"], language="json")
            
            with tabs[3]:
                if "tree_path" in results:
                    try:
                        tree_path = Path(results["tree_path"])
                        if tree_path.exists():
                            with open(tree_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            components.html(
                                html_content,
                                height=800,  # Increased height for better visibility
                                width=None,  # Let it adapt to container width
                                scrolling=True
                            )
                        else:
                            st.error(f"Tree visualization file not found at: {tree_path}")
                    except Exception as e:
                        st.error(f"Error loading tree visualization: {str(e)}")
                        logger.error(f"Tree visualization error: {e}", exc_info=True)
                else:
                    st.info("No tree visualization available for this run.")

if __name__ == "__main__":
    main() 