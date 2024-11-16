import gradio as gr
import pandas as pd
import numpy as np
from pathlib import Path
from ..journal import Journal
from igraph import Graph

def generate_layout(n_nodes, edges, layout_type='rt'):
    """Generate layout coordinates for tree visualization"""
    layout = Graph(n_nodes, edges=edges, directed=True).layout(layout_type)
    y_max = max(layout[k][1] for k in range(n_nodes))
    layout_coords = []
    for n in range(n_nodes):
        layout_coords.append((layout[n][0], 2 * y_max - layout[n][1]))
    return np.array(layout_coords)

def normalize_layout(layout: np.ndarray):
    """Normalize layout coordinates to [0,1] range"""
    # Handle x coordinates
    x_range = layout.max(axis=0)[0] - layout.min(axis=0)[0]
    if x_range == 0:
        layout[:, 0] = 0.5
    else:
        layout[:, 0] = (layout[:, 0] - layout.min(axis=0)[0]) / x_range

    # Handle y coordinates
    y_range = layout.max(axis=0)[1] - layout.min(axis=0)[1]
    if y_range == 0:
        layout[:, 1] = 0.5
    else:
        layout[:, 1] = (layout[:, 1] - layout.min(axis=0)[1]) / y_range
        
    layout[:, 1] = 1 - layout[:, 1]  # Invert y axis
    return layout

def get_edges(journal: Journal):
    """Get edges from journal"""
    return [(node.step, child.step) for node in journal for child in node.children]

def create_tree_visualization(journal: Journal):
    """Create tree visualization using Gradio plots"""
    # Get edges and layout
    edges = list(get_edges(journal))
    layout = normalize_layout(generate_layout(len(journal), edges))
    
    # Create dataframe for nodes
    nodes_df = pd.DataFrame({
        "x": layout[:, 0],
        "y": layout[:, 1],
        "node_id": range(len(journal)),
        "plan": [n.plan for n in journal.nodes],
        "code": [n.code for n in journal.nodes],
        "analysis": [n.analysis for n in journal.nodes]
    })
    
    # Create dataframe for edges
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    
    with gr.Blocks() as demo:
        gr.Markdown("## Solution Tree Visualization")
        
        with gr.Row():
            # Tree visualization
            tree_plot = gr.ScatterPlot(
                nodes_df,
                x="x",
                y="y",
                tooltip=["node_id", "plan"],
                title="Solution Tree",
                height=600,
                width=800
            )
            
            # Node details
            with gr.Column():
                selected_node = gr.Number(label="Selected Node ID", value=-1)
                node_plan = gr.Textbox(label="Plan", lines=3)
                node_code = gr.Code(label="Code", language="python")
                node_analysis = gr.Textbox(label="Analysis", lines=3)
        
        # Add edges as lines
        for _, edge in edges_df.iterrows():
            src, tgt = edge["source"], edge["target"]
            tree_plot.add_trace({
                "type": "scatter",
                "x": [nodes_df.loc[src, "x"], nodes_df.loc[tgt, "x"]],
                "y": [nodes_df.loc[src, "y"], nodes_df.loc[tgt, "y"]],
                "mode": "lines",
                "line": {"color": "gray", "width": 1},
                "showlegend": False
            })
        
        # Handle node selection
        def update_node_info(evt: gr.SelectData):
            node_id = evt.index
            node = journal.nodes[node_id]
            return {
                selected_node: node_id,
                node_plan: node.plan,
                node_code: node.code,
                node_analysis: node.analysis
            }
        
        tree_plot.select(
            update_node_info,
            None,
            [selected_node, node_plan, node_code, node_analysis]
        )
        
    return demo

def launch_tree_viz(journal: Journal):
    """Launch the tree visualization app"""
    demo = create_tree_visualization(journal)
    demo.launch(
        share=False,
        height=800,
        server_port=7861,  # Different port from main app
        show_error=True
    )

if __name__ == "__main__":
    # Example usage with a sample journal
    from ..journal import Journal
    
    # Create sample journal for testing
    journal = Journal()
    # Add some sample nodes and edges
    # ... add your 