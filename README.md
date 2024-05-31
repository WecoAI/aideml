# AIDE: the Machine Learning CodeGen Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)&ensp;
[![PyPI](https://img.shields.io/pypi/v/aideml?color=blue)](https://pypi.org/project/aideml/)&ensp;
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Discord](https://dcbadge.vercel.app/api/server/Rq7t8wnsuA?compact=true&style=flat)](https://discord.gg/Rq7t8wnsuA)&ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/WecoAI?style=social)](https://twitter.com/WecoAI)&ensp;

AIDE is an LLM agent that generates solutions for machine learning tasks just from natural language descriptions of the task. In a benchmark composed of over 60 Kaggle data science competitions, AIDE demonstrated impressive performance, surpassing 50% of Kaggle participants on average (see our [technical report](https://www.weco.ai/blog/technical-report) for details).
More specifically, AIDE has the following features:

1. **Instruct with Natural Language**: Describe your problem or additional requirements and expert insights, all in natural language.
2. **Deliver Solution in Source Code**: AIDE will generate Python scripts for the **tested** machine learning pipeline. Enjoy full transparency, reproducibility, and the freedom to further improve the source code!
3. **Iterative Optimization**: AIDE iteratively runs, debugs, evaluates, and improves the ML code, all by itself.
4. **Visualization**: We also provide tools to visualize the solution tree produced by AIDE for a better understanding of its experimentation process. This gives you insights not only about what works but also what doesn't.

# How to use AIDE?

## Setup

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

## Running AIDE via the command line

To run AIDE:

```bash
aide data_dir="<path to your data directory>" goal="<describe the agent's goal for your task>" eval="<(optional) describe the evaluation metric the agent should use>"
```

For example, to run AIDE on the example [house price prediction task](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data):

```bash
aide data_dir="example_tasks/house_prices" goal="Predict the sales price for each house" eval="Use the RMSE metric between the logarithm of the predicted and observed values."
```

Options:

- `data_dir` (required): a directory containing all the data relevant for your task (`.csv` files, images, etc.).
- `goal`: describe what you want the models to predict in your task, for example, "Build a timeseries forcasting model for bitcoin close price" or "Predict sales price for houses".
- `eval`: the evaluation metric used to evaluate the ML models for the task (e.g., accuracy, F1, Root-Mean-Squared-Error, etc.)

Alternatively, you can provide the entire task description as a `desc_str` string, or write it in a plaintext file and pass its path as `desc_file` ([example file](aide/example_tasks/house_prices.md)).

```bash
aide data_dir="my_data_dir" desc_file="my_task_description.txt"
```

The result of the run will be stored in the `logs` directory.

- `logs/<experiment-id>/best_solution.py`: Python code of _best solution_ according to the validation metric
- `logs/<experiment-id>/journal.json`: a JSON file containing the metadata of the experiment runs, including all the code generated in intermediate steps, plan, evaluation results, etc.
- `logs/<experiment-id>/tree_plot.html`: you can open it in your browser. It contains visualization of solution tree, which details the experimentation process of finding and optimizing ML code. You can explore and interact with the tree visualization to view what plan and code AIDE comes up with in each step.

The `workspaces` directory will contain all the files and data that the agent generated.

### Advanced Usage

To further customize the behaviour of AIDE, some useful options might be:

- `agent.code.model=...` to configure which model the agent should use for coding (default is `gpt-4-turbo`)
- `agent.steps=...` to configure how many improvement iterations the agent should run (default is 20)
- `agent.search.num_drafts=...` to configure the number of initial drafts the agent should generate (default is 5)

You can check the [`config.yaml`](aide/utils/config.yaml) file for more options.

## Using AIDE in Python

Using AIDE within your Python script/project is easy. Follow the setup steps above, and then create an AIDE experiment like below and start running:

```python
import aide
exp = aide.Experiment(
    data_dir="example_tasks/bitcoin_price",  # replace this with your own directory
    goal="Build a timeseries forcasting model for bitcoin close price.",  # replace with your own goal description
    eval="RMSLE"  # replace with your own evaluation metric
)

best_solution = exp.run(steps=10)

print(f"Best solution has validation metric: {best_solution.valid_metric}")
print(f"Best solution code: {best_solution.code}")
```

## Development

To install AIDE for development, clone this repository and install it locally.

```bash
git clone https://github.com/WecoAI/aideml.git
cd aideml
pip install -e .
```

Contribution guide will be available soon.

## Algorithm Description

AIDE's problem-solving approach is inspired by how human data scientists tackle challenges. It starts by generating a set of initial solution drafts and then iteratively refines and improves them based on performance feedback. This process is driven by a technique we call Solution Space Tree Search.

At its core, Solution Space Tree Search consists of three main components:

- **Solution Generator**: This component proposes new solutions by either creating novel drafts or making changes to existing solutions, such as fixing bugs or introducing improvements.
- **Evaluator**: The evaluator assesses the quality of each proposed solution by running it and comparing its performance against the objective. This is implemented by instructing the LLM to include statements that print the evaluation metric and by having another LLM parse the printed logs to extract the evaluation metric.
- **Base Solution Selector**: The solution selector picks the most promising solution from the explored options to serve as the starting point for the next iteration of refinement.

By repeatedly applying these steps, AIDE navigates the vast space of possible solutions, progressively refining its approach until it converges on the optimal solution for the given data science problem.

![Tree Search Visualization](https://github.com/WecoAI/aideml/assets/8918572/2401529c-b97e-4029-aed2-c3f376f54c3c)
