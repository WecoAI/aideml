import gradio as gr
from pathlib import Path
import tempfile
import shutil

from aide.webui.tree_viz import create_tree_visualization
from .. import Experiment
from ..utils.tree_export import generate_html, cfg_to_tree_struct, generate as tree_export_generate
from rich.console import Console
import sys
from omegaconf import OmegaConf
from ..utils.config import load_task_desc, prep_agent_workspace, save_run, prep_cfg
from ..journal import Journal
from ..agent import Agent
from ..interpreter import Interpreter
import json
from .. import run as aide_run
import os
from rich.markdown import Markdown
from rich import print_json
import time
import json
import textwrap
import numpy as np
from igraph import Graph
import webbrowser
from .tree_viz import create_tree_visualization

from aide import journal

console = Console(file=sys.stderr)

def load_example_task():
    """Load the house prices example task"""
    example_dir = Path(__file__).parent.parent / "example_tasks" / "house_prices"
    
    # Just return the file paths directly
    temp_files = []
    for file_path in example_dir.glob("*"):
        if file_path.suffix.lower() in ['.csv', '.txt', '.json', '.md']:
            temp_files.append(str(file_path))  # Convert Path to string
    
    example_goal = "Predict the sales price for each house"
    example_eval = "Use the RMSE metric between the logarithm of the predicted and observed values."
    return temp_files, example_goal, example_eval, 10

def load_previous_experiment(exp_dir: str):
    """Load results from a previous experiment directory"""
    try:
        exp_path = Path(exp_dir)
        if not exp_path.exists():
            return f"Error: Directory {exp_dir} does not exist", None, None, None
            
        # Load the best solution file
        solution_path = exp_path / "best_solution.py"
        if not solution_path.exists():
            return "Error: No best solution file found in experiment directory", None, None, None
            
        # Load config file
        config_path = exp_path / "config.yaml"
        if not config_path.exists():
            return "Error: No config file found in experiment directory", None, None, None
        cfg = OmegaConf.load(config_path)
        
        # Load journal file
        journal_path = exp_path / "journal.json"
        if not journal_path.exists():
            return "Error: No journal file found in experiment directory", None, None, None
        
        with open(journal_path, 'r') as f:
            journal_data = json.load(f)
            
        # Get tree visualization file path
        tree_path = exp_path / "tree_plot.html"
        # Use relative path from project root
        relative_path = tree_path.relative_to(Path(__file__).parent.parent.parent)
        tree_link = f"""
        <a href="javascript:void(0)" 
           onclick="fetch('/open-file?path={tree_path.absolute()}')" 
           style="text-decoration: underline; color: blue; cursor: pointer;">
           Open Tree Visualization
        </a>
        """
        
        # Add route to open file
        @demo.app.get("/open-file")
        def open_file(path: str):
            webbrowser.open(f'file://{path}')
            return {"status": "success"}
        
        return (
            solution_path.read_text(),
            OmegaConf.to_yaml(cfg),
            json.dumps(journal_data, indent=2),
            tree_link
        )
        
    except Exception as e:
        console.print_exception()
        return f"Error loading experiment: {str(e)}", None, None, None

def create_ui():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Centered title and description
        gr.Markdown("""
            <div style="text-align: center; margin-bottom: 2em;">
                <h1 style="margin-bottom: 0.5em;">AIDE: AI Development Environment</h1>
                <p>An LLM agent that generates solutions for machine learning tasks from natural language descriptions.</p>
            </div>
        """)
        
        with gr.Row():
            # Left column for configuration
            with gr.Column(scale=1):
                with gr.Accordion("Configuration", open=True):
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        value=os.getenv("OPENAI_API_KEY", "")
                    )
                    anthropic_api_key = gr.Textbox(
                        label="Anthropic API Key (Optional)",
                        type="password",
                        value=os.getenv("ANTHROPIC_API_KEY", "")
                    )
                    data_dir = gr.File(
                        label="Data Files",
                        file_count="multiple",
                        file_types=[".csv", ".txt", ".json", ".md"]
                    )
                    goal_text = gr.Textbox(
                        label="Goal",
                        placeholder="Example: Predict house prices",
                        lines=3
                    )
                    eval_text = gr.Textbox(
                        label="Evaluation Criteria",
                        placeholder="Example: Use RMSE metric",
                        lines=2
                    )
                    num_steps = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Number of Steps"
                    )
                    
                    # Add experiment loading section
                    with gr.Accordion("Load Previous Experiment", open=False):
                        exp_dir = gr.Textbox(
                            label="Experiment Directory",
                            placeholder="Path to experiment directory (e.g., logs/exp_20240321_123456)",
                            value="/Users/dex/Work/wecoai/aideml/logs/2-remarkable-aloof-llama",
                            lines=1
                        )
                        load_exp_btn = gr.Button("Load Experiment")
                    
                    with gr.Row():
                        load_example_btn = gr.Button("Load Example")
                        run_btn = gr.Button("Run AIDE", variant="primary")

            # Right column for results
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Best Solution"):
                        output_code = gr.Code(
                            label="Best Solution",
                            language="python",
                            show_label=True
                        )
                    with gr.TabItem("Config"):
                        config_output = gr.Code(
                            label="Configuration",
                            language="yaml",
                            show_label=True
                        )
                    with gr.TabItem("Journal"):
                        journal_output = gr.Code(
                            label="Journal",
                            language="json",
                            show_label=True
                        )
                    with gr.TabItem("Tree Visualization"):
                        tree_viz = create_tree_visualization(journal)  # Add when journal is available

        # Connect the buttons
        run_btn.click(
            fn=run_aide,
            inputs=[openai_api_key, anthropic_api_key, data_dir, goal_text, eval_text, num_steps],
            outputs=[output_code, config_output, journal_output, tree_viz]
        )

        load_example_btn.click(
            fn=load_example_task,
            inputs=[],
            outputs=[data_dir, goal_text, eval_text, num_steps]
        )
        
        # Add load experiment button handler
        load_exp_btn.click(
            fn=load_previous_experiment,
            inputs=[exp_dir],
            outputs=[output_code, config_output, journal_output, tree_output]
        )

    return demo

def generate_layout(n_nodes, edges, layout_type='rt'):
    layout = Graph(n_nodes, edges=edges, directed=True).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)

def normalize_layout(layout: np.ndarray):
    """Normalize layout coordinates while handling edge cases"""
    # Handle x coordinates
    x_range = layout.max(axis=0)[0] - layout.min(axis=0)[0]
    if x_range == 0:
        layout[:, 0] = 0.5  # Center horizontally if all x values are same
    else:
        layout[:, 0] = (layout[:, 0] - layout.min(axis=0)[0]) / x_range

    # Handle y coordinates
    y_range = layout.max(axis=0)[1] - layout.min(axis=0)[1]
    if y_range == 0:
        layout[:, 1] = 0.5  # Center vertically if all y values are same
    else:
        layout[:, 1] = (layout[:, 1] - layout.min(axis=0)[1]) / y_range
        
    layout[:, 1] = 1 - layout[:, 1]  # Invert y axis
    return layout

def cfg_to_tree_struct(cfg, journal):
    edges = [(node.step, child.step) for node in journal for child in node.children]
    layout = normalize_layout(generate_layout(len(journal), edges))
    metrics = np.array([0 for _ in journal])
    tree_struct = {
        'edges': edges,
        'layout': layout.tolist(),
        'plan': [textwrap.fill(node.plan, width=80) for node in journal.nodes],
        'code': [node.code for node in journal],
        'term_out': [node.term_out for node in journal],
        'analysis': [node.analysis for node in journal],
        'exp_name': cfg.exp_name,
        'metrics': metrics.tolist(),
    }
    return tree_struct


def create_tree_component():
    return gr.HTML(
        value="",
        elem_classes=["tree-container"],
        elem_id="tree-viz"
    )

def run_aide(openai_api_key, anthropic_api_key, files, goal_text, eval_text, num_steps, progress=gr.Progress(track_tqdm=True)):
    try:
        # Set API keys
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

        # Create input directory in the project root
        project_root = Path(__file__).parent.parent.parent
        input_dir = project_root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
            
        # Handle uploaded files
        if files:
            for file in files:
                shutil.copy2(file.name, input_dir)
        else:
            return "Error: Please upload data files", None, None, None

        # Load and prepare config
        cfg = OmegaConf.load(Path(__file__).parent.parent / "utils" / "config.yaml")
        cfg.data_dir = str(input_dir)
        cfg.goal = goal_text
        cfg.eval = eval_text
        cfg.agent.steps = num_steps
        
        cfg = prep_cfg(cfg)
        task_desc = load_task_desc(cfg)
        prep_agent_workspace(cfg)

        # Initialize components
        journal = Journal()
        agent = Agent(task_desc=task_desc, cfg=cfg, journal=journal)
        interpreter = Interpreter(str(cfg.workspace_dir), **OmegaConf.to_container(cfg.exec))

        # Execute steps with progress tracking
        for step in range(num_steps):
            progress(step/num_steps, f"Step {step + 1}/{num_steps}")
            agent.step(exec_callback=interpreter.run)
            save_run(cfg, journal)

        progress(1.0, "Complete!")

        # Read results
        solution_path = cfg.workspace_dir / "best_solution.py"
        solution_code = solution_path.read_text() if solution_path.exists() else "No solution generated"
        
        # Save tree visualization to file and get path
        tree_struct = cfg_to_tree_struct(cfg, journal)
        tree_path = cfg.workspace_dir / "tree_plot.html"
        tree_export_generate(tree_struct, str(tree_path))
        
        return (
            solution_code,
            OmegaConf.to_yaml(cfg),
            json.dumps(journal.to_dict(), indent=2),
            str(tree_path.absolute())  # Return absolute path as string
        )

    except Exception as e:
        console.print_exception()
        return f"Error occurred: {str(e)}", None, None, None

def handle_tree_update(html_content):
    """Handle tree visualization updates"""
    # Return empty values since the actual updates are handled by JavaScript
    return "", ""

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        share=False,  # Enable sharing
        height=900,  # Ensure enough height for visualization
        server_port=7860,  # Specify a port
        allowed_paths=[str(Path(__file__).parent.parent.parent / "logs")],  # Allow serving files from logs directory
        show_error=True
    )