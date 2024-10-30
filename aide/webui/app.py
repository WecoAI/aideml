import gradio as gr
from pathlib import Path
import tempfile
import shutil
from .. import Experiment
from ..utils.tree_export import generate_html, cfg_to_tree_struct
from rich.console import Console
import sys
from omegaconf import OmegaConf
from ..utils.config import load_task_desc, prep_agent_workspace, save_run
from ..journal import Journal
from ..agent import Agent
from ..interpreter import Interpreter
import json
from .. import run as aide_run
import os
from rich.markdown import Markdown
from rich import print_json
import time

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

def create_ui():
    # Add custom CSS to handle iframe sizing and styling
    custom_css = """
        <style>
            .tree-viz-container {
                width: 100%;
                height: 800px;
                margin: 0;
                padding: 0;
                background: #f2f0e7;
                overflow: hidden;
            }
            .tree-viz-iframe {
                width: 100%;
                height: 100%;
                border: none;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }
        </style>
    """
    
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown("""
            <div style="text-align: center; margin-bottom: 2em;">
                <h1>AIDE: AI Development Environment</h1>
                <p>An LLM agent that generates solutions for machine learning tasks from natural language descriptions.</p>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Add configuration accordion
                with gr.Accordion("API Configuration", open=False):
                    openai_api_key = gr.Textbox(
                        label="OpenAI API Key", 
                        placeholder="sk-...",
                        type="password",
                        value=os.getenv("OPENAI_API_KEY", "")
                    )
                    openai_base_url = gr.Textbox(
                        label="OpenAI Base URL (Optional)",
                        placeholder="https://api.openai.com/v1",
                        value=os.getenv("OPENAI_BASE_URL", ""),
                        visible=True
                    )
                    anthropic_api_key = gr.Textbox(
                        label="Anthropic API Key (Optional)",
                        placeholder="sk-ant-...", 
                        type="password",
                        value=os.getenv("ANTHROPIC_API_KEY", "")
                    )

                data_dir = gr.File(
                    label="Data Files", 
                    file_count="multiple",
                    file_types=[".csv", ".txt", ".json", ".md"],
                    type="filepath"
                )
                
                goal_text = gr.Textbox(
                    label="Goal", 
                    placeholder="Example: Build a timeseries forecasting model for bitcoin close price",
                    lines=3
                )
                
                eval_text = gr.Textbox(
                    label="Evaluation Criteria (Optional)", 
                    placeholder="Example: Use RMSE metric between the logarithm of predicted and observed values",
                    lines=2
                )
                
                num_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="Number of Steps"
                )
                
                with gr.Row():
                    load_example_btn = gr.Button("Load Example", variant="secondary")
                    run_btn = gr.Button("Run AIDE", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.Tab("Code"):
                        output_code = gr.Code(language="python", label="Generated Code")
                        output_metrics = gr.JSON(label="Metrics")
                        output_status = gr.Textbox(label="Status")
                    
                    with gr.Tab("Solution Tree"):
                        output_tree = gr.HTML(
                            label="Solution Tree",
                            value="<div style='height: 800px; background: #f2f0e7;'>Tree visualization will appear here...</div>"
                        )

        # Event handlers
        load_example_btn.click(
            fn=load_example_task,
            inputs=[],
            outputs=[data_dir, goal_text, eval_text, num_steps]
        )
        
        run_btn.click(
            fn=run_aide,
            inputs=[
                openai_api_key,
                openai_base_url,
                anthropic_api_key,
                data_dir,
                goal_text,
                eval_text,
                num_steps
            ],
            outputs=[output_code, output_metrics, output_tree, output_status],
            queue=True,
            api_name="run_aide"
        )

    return demo

def wrap_tree_html(tree_html: str) -> str:
    
    """Wrap the tree visualization HTML with proper iframe setup"""
    # Add base target to prevent iframe navigation issues
    wrapped_html = tree_html.replace('<head>', '<head><base target="_parent">')
    
    # Ensure proper iframe setup with required attributes
    return f"""
        <iframe
            id="tree-viz-frame"
            srcdoc='{wrapped_html}'
            style="width: 100%; height: 800px; border: none; background: #f2f0e7;"
            sandbox="allow-scripts allow-same-origin"
        ></iframe>
    """

def run_aide(openai_api_key, openai_base_url, anthropic_api_key, files, goal_text, eval_text, num_steps, progress=gr.Progress(track_tqdm=True)):
    try:
        # Set API keys and base URL
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            # Validate OpenAI key format
            if not openai_api_key.startswith('sk-'):
                raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-'")
                
        if openai_base_url:
            os.environ["OPENAI_BASE_URL"] = openai_base_url
            
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            # Validate Anthropic key format
            if not anthropic_api_key.startswith('sk-ant-'):
                raise ValueError("Invalid Anthropic API key format. Key should start with 'sk-ant-'")

        if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
            raise ValueError("Either OpenAI or Anthropic API key is required. Please provide at least one in the API Configuration section.")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Load and override config
            config_path = Path(__file__).parent.parent / "utils" / "config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at: {config_path}")
                
            cfg = OmegaConf.load(config_path)
            
            # Create necessary directories first
            workspace_dir = Path(temp_dir) / "workspace"
            log_dir = Path(temp_dir) / "logs"
            workspace_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            
            # Set up configuration
            cfg.data_dir = str(Path(temp_dir) / "input")
            cfg.goal = goal_text
            cfg.eval = eval_text if eval_text else None
            cfg.exp_name = "gradio_run"
            cfg.agent.steps = num_steps
            cfg.workspace_dir = str(workspace_dir)
            cfg.log_dir = str(log_dir)
            
            # Create data directory and copy files
            data_dir = Path(cfg.data_dir)
            data_dir.mkdir(parents=True)
            
            if files:
                for file in files:
                    src_path = file.get('name', file) if isinstance(file, dict) else file
                    
                    if not isinstance(src_path, (str, Path)):
                        continue
                        
                    src_path = Path(src_path)
                    if src_path.suffix.lower() not in ['.csv', '.txt', '.json', '.md']:
                        continue
                        
                    dest_path = data_dir / src_path.name
                    shutil.copy2(src_path, dest_path)
            
            # Initialize components
            task_desc = load_task_desc(cfg)
            journal = Journal()
            agent = Agent(task_desc=task_desc, cfg=cfg, journal=journal)
            interpreter = Interpreter(
                cfg.workspace_dir, 
                **OmegaConf.to_container(cfg.exec)
            )

            # Execute steps with live updates
            progress(0, desc="Running AIDE (0%)")
            best_node = None
            metric_value = 0.0  # Initialize metric_value
            tree_html = ""  # Initialize tree_html
            
            for step in range(num_steps):
                percentage = ((step + 1) / num_steps) * 100
                progress((step + 1) / num_steps, desc=f"Running AIDE ({percentage:.0f}%)")
                console.print(f"\nStarting step {step + 1}/{num_steps}")
                
                # Run the agent step
                agent.step(exec_callback=interpreter.run)
                
                # Get results
                best_node = journal.get_best_node(only_good=False)
                tree_html = generate_html(json.dumps(cfg_to_tree_struct(cfg, journal)))
                
                if best_node and best_node.metric:
                    try:
                        if hasattr(best_node.metric, 'value'):
                            metric_value = float(best_node.metric.value) if best_node.metric.value is not None else 0.0
                        elif isinstance(best_node.metric, (int, float)):
                            metric_value = float(best_node.metric)
                    except (ValueError, TypeError) as e:
                        console.print(f"[yellow]Warning: Could not convert metric to float: {e}[/yellow]")
                        metric_value = 0.0
                
                console.print(f"Step {step + 1} completed successfully")
                
                # Update the UI with current values
                current_code = best_node.code if best_node else ""
                current_metrics = {"validation_metric": metric_value}
                current_tree = wrap_tree_html(tree_html)
                current_status = f"Completed step {step + 1}/{num_steps}"
                
                yield current_code, current_metrics, current_tree, current_status

            console.print("\n[green]All steps completed![/green]")
            interpreter.cleanup_session()

    except Exception as e:
        console.print("[red]Fatal error:[/red]")
        console.print_exception()
        yield None, None, None, f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        share=True,  # Enable sharing
        height=900,  # Ensure enough height for visualization
        server_port=7860  # Specify a port
    )