<h1 align="center">AIDE: The Machine Learning Engineer Agent</h1>

<p align="center">
          📑 <a href="https://arxiv.org/abs/2502.13138">Paper</a>&nbsp&nbsp | &nbsp&nbsp📝 <a href="https://www.weco.ai/blog/technical-report">Blog</a>&nbsp&nbsp | &nbsp&nbsp🌐 <a href="https://www.aide.ml">Project</a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pypi.org/project/aideml/"><img src="https://img.shields.io/pypi/v/aideml?color=blue" alt="PyPI"></a>&ensp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>&ensp;
  <a href="https://twitter.com/WecoAI"><img src="https://img.shields.io/twitter/follow/WecoAI?style=social" alt="Twitter Follow"></a>&ensp;
</p>

AIDE is an LLM agent that generates solutions for machine learning tasks just from natural language descriptions of the task.

In our own benchmark composed of over 60 Kaggle data science competitions, AIDE demonstrated impressive performance, surpassing 50% of Kaggle participants on average.

OpenAI's [MLE-bench](https://arxiv.org/pdf/2410.07095), a benchmark composed of 75 Kaggle machine learning tasks, shows that AIDE achieved four times more medals compared to the runner-up agent architecture.

METR's [RE-Bench](https://arxiv.org/pdf/2411.15114) shows that AIDE is not only capable at machine learning tasks but generalizes to the AI R&D tasks such as optimizing low level Triton kernels and finetuning GPT-2 for QA, even surpassing the performance of human experts.

More specifically, AIDE has the following features:

1. **Instruct with Natural Language**: Describe your problem or additional requirements and expert insights, all in natural language.
2. **Deliver Solution in Source Code**: AIDE will generate Python scripts for the **tested** machine learning pipeline. Enjoy full transparency, reproducibility, and the freedom to further improve the source code!
3. **Iterative Optimization**: AIDE iteratively runs, debugs, evaluates, and improves the ML code, all by itself.
4. **Visualization**: We also provide tools to visualize the solution tree produced by AIDE for a better understanding of its experimentation process. This gives you insights not only about what works but also what doesn't.

# How to Use AIDE?

## Running AIDE via the Web UI


https://github.com/user-attachments/assets/1da42853-fe36-45e1-b6a2-852f88470af6


We have developed a user-friendly Web UI using Streamlit to make it even easier to interact with AIDE.

### Prerequisites

Ensure you have installed the development version of AIDE and its dependencies as described in the [Development](#development) section.

### Running the Web UI

Navigate to the `aide/webui` directory and run the Streamlit application:

```bash
cd aide/webui
streamlit run app.py
```

Alternatively, you can run it from the root directory:

```bash
streamlit run aide/webui/app.py
```

### Using the Web UI

1. **API Key Configuration**: In the sidebar, input your OpenAI API key or Anthropic API key and click "Save API Keys".

2. **Input Data**:
   - You can either **upload your dataset files** (`.csv`, `.txt`, `.json`, `.md`) using the "Upload Data Files" feature.
   - Or click on "Load Example Experiment" to use the example house prices dataset.

3. **Define Goal and Evaluation Criteria**:
   - In the "Goal" text area, describe what you want the model to achieve (e.g., "Predict the sales price for each house").
   - In the "Evaluation Criteria" text area, specify the evaluation metric (e.g., "Use the RMSE metric between the logarithm of the predicted and observed values.").

4. **Configure Steps**:
   - Use the slider to set the number of steps (iterations) for the experiment.

5. **Run the Experiment**:
   - Click on "Run AIDE" to start the experiment.
   - Progress and status updates will be displayed in the "Results" section.

6. **View Results**:
   - **Tree Visualization**: Explore the solution tree to understand how AIDE experimented and optimized the models.
   - **Best Solution**: View the Python code of the best solution found.
   - **Config**: Review the configuration used for the experiment.
   - **Journal**: Examine the detailed journal entries for each step.


## Running AIDE via the Command Line

### Setup

Make sure you have `Python>=3.10` installed and run:

```bash
pip install -U aideml
```

Also install `unzip` to allow the agent to autonomously extract your data.

Set up your OpenAI (or Anthropic) API key:

```bash
export OPENAI_API_KEY=<your API key>
# or
export ANTHROPIC_API_KEY=<your API key>
```

To run AIDE:

```bash
aide data_dir="<path to your data directory>" goal="<describe the agent's goal for your task>" eval="<(optional) describe the evaluation metric the agent should use>"
```

For example, to run AIDE on the example [house price prediction task](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data):

```bash
aide data_dir="example_tasks/house_prices" goal="Predict the sales price for each house" eval="Use the RMSE metric between the logarithm of the predicted and observed values."
```

Options:

- `data_dir` (required): A directory containing all the data relevant for your task (`.csv` files, images, etc.).
- `goal`: Describe what you want the models to predict in your task, for example, "Build a time series forecasting model for bitcoin close price" or "Predict sales price for houses".
- `eval`: The evaluation metric used to evaluate the ML models for the task (e.g., accuracy, F1, Root-Mean-Squared-Error, etc.).

Alternatively, you can provide the entire task description as a `desc_str` string, or write it in a plaintext file and pass its path as `desc_file` ([example file](aide/example_tasks/house_prices.md)).

```bash
aide data_dir="my_data_dir" desc_file="my_task_description.txt"
```

The result of the run will be stored in the `logs` directory.

- `logs/<experiment-id>/best_solution.py`: Python code of the _best solution_ according to the validation metric.
- `logs/<experiment-id>/journal.json`: A JSON file containing the metadata of the experiment runs, including all the code generated in intermediate steps, plan, evaluation results, etc.
- `logs/<experiment-id>/tree_plot.html`: You can open it in your browser. It contains a visualization of the solution tree, which details the experimentation process of finding and optimizing ML code. You can explore and interact with the tree visualization to view what plan and code AIDE comes up with in each step.

The `workspaces` directory will contain all the files and data that the agent generated.

### Advanced Usage

To further customize the behavior of AIDE, some useful options might be:

- `agent.code.model=...` to configure which model the agent should use for coding (default is `gpt-4-turbo`).
- `agent.steps=...` to configure how many improvement iterations the agent should run (default is 20).
- `agent.search.num_drafts=...` to configure the number of initial drafts the agent should generate (default is 5).

You can check the [`config.yaml`](aide/utils/config.yaml) file for more options.

### Using Local LLMs

AIDE supports using local LLMs through OpenAI-compatible APIs. Here's how to set it up:

1. Set up a local LLM server with an OpenAI-compatible API endpoint. You can use:
   - [Ollama](https://github.com/ollama/ollama)
   - or similar solutions.

2. Configure your environment to use the local endpoint:

   ```bash
   export OPENAI_BASE_URL="http://localhost:11434/v1"  # For Ollama
   export OPENAI_API_KEY="local-llm"  # Can be any string if your local server doesn't require authentication
   ```

3. Update the model configuration in your AIDE command or config. For example, with Ollama:

   ```bash
   # Example with house prices dataset
   aide agent.code.model="qwen2.5" agent.feedback.model="qwen2.5" report.model="qwen2.5" \
       data_dir="example_tasks/house_prices" \
       goal="Predict the sales price for each house" \
       eval="Use the RMSE metric between the logarithm of the predicted and observed values."
   ```

## Using AIDE in Python

Using AIDE within your Python script/project is easy. Follow the setup steps above, and then create an AIDE experiment like below and start running:

```python
import aide
exp = aide.Experiment(
    data_dir="example_tasks/bitcoin_price",  # replace this with your own directory
    goal="Build a time series forecasting model for bitcoin close price.",  # replace with your own goal description
    eval="RMSLE"  # replace with your own evaluation metric
)

best_solution = exp.run(steps=10)

print(f"Best solution has validation metric: {best_solution.valid_metric}")
print(f"Best solution code: {best_solution.code}")
```

## Development

To install AIDE for development, clone this repository and install it locally:

```bash
git clone https://github.com/WecoAI/aideml.git
cd aideml
pip install -e .
```

### Running the Web UI in Development Mode

Ensure that you have all the required development dependencies installed. Then, you can run the Web UI as follows:

```bash
cd aide/webui
streamlit run app.py
```

## Using AIDE with Docker

You can also run AIDE using Docker:

1. **Build the Docker Image**:

   ```bash
   docker build -t aide .
   ```

2. **Run AIDE with Docker** (example with house prices task):

   ```bash
   # Set custom workspace and logs location (optional)
   export WORKSPACE_BASE=$(pwd)/workspaces
   export LOGS_DIR=$(pwd)/logs

   docker run -it --rm \
             -v "${LOGS_DIR:-$(pwd)/logs}:/app/logs" \
             -v "${WORKSPACE_BASE:-$(pwd)/workspaces}:/app/workspaces" \
             -v "$(pwd)/aide/example_tasks:/app/data" \
             -e OPENAI_API_KEY="your-actual-api-key" \
             aide \
             data_dir=/app/data/house_prices \
             goal="Predict the sales price for each house" \
             eval="Use the RMSE metric between the logarithm of the predicted and observed values."
   ```

You can customize the location of workspaces and logs by setting environment variables before running the container:

- `WORKSPACE_BASE`: Sets the base directory for AIDE workspaces (default: `$(pwd)/workspaces`)
- `LOGS_DIR`: Sets the directory for AIDE logs (default: `$(pwd)/logs`)

Contribution guide will be available soon.

## Algorithm Description

AIDE's problem-solving approach is inspired by how human data scientists tackle challenges. It starts by generating a set of initial solution drafts and then iteratively refines and improves them based on performance feedback. This process is driven by a technique we call Solution Space Tree Search.

At its core, Solution Space Tree Search consists of three main components:

- **Solution Generator**: This component proposes new solutions by either creating novel drafts or making changes to existing solutions, such as fixing bugs or introducing improvements.
- **Evaluator**: The evaluator assesses the quality of each proposed solution by running it and comparing its performance against the objective. This is implemented by instructing the LLM to include statements that print the evaluation metric and by having another LLM parse the printed logs to extract the evaluation metric.
- **Base Solution Selector**: The solution selector picks the most promising solution from the explored options to serve as the starting point for the next iteration of refinement.

By repeatedly applying these steps, AIDE navigates the vast space of possible solutions, progressively refining its approach until it converges on the optimal solution for the given data science problem.

![Tree Search Visualization](https://github.com/WecoAI/aideml/assets/8918572/2401529c-b97e-4029-aed2-c3f376f54c3c)

# Citation

If you use AIDE in your work, please cite the following paper:
```bibtex
@article{aide2025,
      title={AIDE: AI-Driven Exploration in the Space of Code}, 
      author={Zhengyao Jiang and Dominik Schmidt and Dhruv Srikanth and Dixing Xu and Ian Kaplan and Deniss Jacenko and Yuxiang Wu},
      year={2025},
      eprint={2502.13138},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.13138}, 
}
```
