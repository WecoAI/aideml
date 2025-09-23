<h1 align="center">AIDE ML — The Machine Learning Engineering Agent</h1>

<p align="center"><em>
LLM‑driven agent that writes, evaluates & improves machine‑learning code.
</em></p>

<p align="center">
<a href="https://pypi.org/project/aideml/"><img src="https://img.shields.io/pypi/v/aideml?label=PyPI&logo=pypi" alt="PyPI"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+"></a>
<a href="https://arxiv.org/abs/2502.13138"><img src="https://img.shields.io/badge/arXiv-2502.13138-b31b1b?logo=arxiv&logoColor=white" alt="arXiv paper"></a>
<img src="https://img.shields.io/github/license/WecoAI/aideml?color=brightgreen" alt="MIT License">
<a href="https://pepy.tech/projects/aideml"><img src="https://static.pepy.tech/badge/aideml" alt="PyPI Downloads"></a>&ensp;
</p>

<p align="center">
<a href="https://docs.weco.ai/cli/getting-started?utm_source=aidemlrepo" target="_blank"><strong>Use in Production? Try Weco →</strong></a>
</p>

# What Is AIDE ML?

**AIDE ML is the open‑source “reference build” of the AIDE algorithm**, a tree‑search agent that autonomously drafts, debugs and benchmarks code until a user‑defined metric is maximised (or minimised). It ships as a *research‑friendly* Python package with batteries‑included utilities (CLI, visualisation, config presets) so that academics and engineer‑researchers can **replicate the paper, test new ideas, or prototyping ML pipelines**.

![Tree Search Visualization](https://github.com/WecoAI/aideml/assets/8918572/2401529c-b97e-4029-aed2-c3f376f54c3c)

| Layer | Description | Where to find it |
| --- | --- | --- |
| **AIDE *algorithm*** | LLM‑guided agentic tree search in the space of code. | Described in our [paper](https://arxiv.org/abs/2502.13138). |
| **AIDE ML *repo* (this repo)** | Lean implementation for experimentation & extension. | `pip install aideml` |
| **Weco *product*** | The platform generalizes AIDE's capabilities to broader code optimization scenarios, providing experiment tracking and enhanced user control. | [weco.ai](https://weco.ai?utm_source=aidemlrepo) |

### Who should use it?

- **Agent‑architecture researchers** – swap in new search heuristics, evaluators or LLM back‑ends.
- **ML Practitioners** – quickly build a high performance ML pipelines given a dataset.

# Key Capabilities

- **Natural‑language task specification**  Point the agent at a dataset and describe *goal* + *metric* in plain English. No YAML grids or bespoke wrappers.  `aide data_dir=…  goal="Predict churn"  eval="AUROC"` 
- **Iterative *agentic tree search*** Each python script becomes a node in a solution tree; LLM‑generated patches spawn children; metric feedback prunes and guides the search. OpenAI’s **[MLE‑Bench](https://arxiv.org/abs/2410.07095)** (75 Kaggle comps) found the tree‑search of AIDE wins **4 × more medals** than the best linear agent (OpenHands). 

<div align="center">
<img src="https://github.com/user-attachments/assets/a48aa65e-360d-4d91-b4ad-98b0fe2585d4" width="80%">
</div>

<details>
<summary>Utility features provided by this repo</summary>

- **HTML visualiser** – inspect the full solution tree and code attached to each node.
- **Streamlit UI** – prototype ML solution .
- **Model‑neutral plumbing** – OpenAI, Anthropic, Gemini, or any local LLM that speaks the OpenAI API.

</details>

## Featured Research built on/with AIDE

| Institution | Paper / Project Name | Links |
|-------------|----------------------|-------|
| **OpenAI** | MLE-bench: Evaluating Machine-Learning Agents on Machine-Learning Engineering | [Paper](https://arxiv.org/abs/2410.07095), [GitHub](https://github.com/openai/mle-bench) |
| **METR** | RE-Bench: Evaluating frontier AI R&D capabilities of language-model agents against human experts | [Paper](https://arxiv.org/abs/2411.15114), [GitHub](https://github.com/METR/RE-Bench) |
| **Sakana AI** | The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search | [Paper](https://arxiv.org/abs/2504.08066), [GitHub](https://github.com/SakanaAI/AI-Scientist-v2) |
| **Meta** | The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements | [Paper](https://arxiv.org/abs/2506.22419), [GitHub](https://github.com/facebookresearch/llm-speedrunner) |
| **Meta** | AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench | [Paper](https://arxiv.org/abs/2507.02554), [GitHub](https://github.com/facebookresearch/aira-dojo) |
| **SJTU** | ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning | [Paper](https://arxiv.org/abs/2506.16499), [GitHub](https://github.com/sjtu-sai-agents/ML-Master) |

> *Know another public project that cites or forks AIDE?  
> [Open a PR](https://github.com/WecoAI/aideml/pulls) and add it to the table!*


# How to Use AIDE ML

## Quick Start

```bash
# 1  Install
pip install -U aideml

# 2  Set an LLM key
export OPENAI_API_KEY=<your‑key>  # https://platform.openai.com/api-keys

# 3  Run an optimisation
aide data_dir="example_tasks/house_prices" \
     goal="Predict the sales price for each house" \
     eval="RMSE between log‑prices"
```

After the run finishes you’ll find:

- `logs/<id>/best_solution.py` – best code found
- `logs/<id>/tree_plot.html` – click to inspect the solution tree

---

## Web UI

```bash
pip install -U aideml   # adds streamlit
cd aide/webui
streamlit run app.py
```

Use the sidebar to paste your API key, upload data, set **Goal** & **Metric**, then press **Run AIDE**.

The UI shows live logs, the solution tree, and the best code.

---

## Advanced CLI Options

```bash
# Choose a different coding model and run 50 steps
aide agent.code.model="claude-4-sonnet" \
     agent.steps=50 \
     data_dir=… goal=… eval=…
```

Common flags

| Flag | Purpose | Default |
| --- | --- | --- |
| `agent.code.model` | LLM used to write code | `gpt-4-turbo` |
| `agent.steps` | Improvement iterations | `20` |
| `agent.search.num_drafts` | Drafts per step | `5` |

---

## Use AIDE ML Inside Python

```python
import aide
import logging

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    aide_logger = logging.getLogger("aide")
    aide_logger.setLevel(logging.INFO)
    print("Starting experiment...")
    exp = aide.Experiment(
        data_dir="example_tasks/bitcoin_price",  # replace this with your own directory
        goal="Build a time series forecasting model for bitcoin close price.",  # replace with your own goal description
        eval="RMSLE"  # replace with your own evaluation metric
    )

    best_solution = exp.run(steps=2)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")
    print("Experiment finished.")

if __name__ == '__main__':
    main()
```

---

## Power‑User Extras

### Local LLM (Ollama example)

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
aide agent.code.model="qwen2.5" data_dir=… goal=… eval=…
```

Note: evaluator defaults to gpt‑4o.

### Fully local (code + evaluator — no external calls)
```
export OPENAI_BASE_URL="http://localhost:11434/v1"
aide agent.code.model="qwen2.5" agent.feedback.model="qwen2.5" data_dir=… goal=… eval=…
```

Tip: Expect some performance drop with fully local models.

### Docker

```bash
docker build -t aide .
docker run -it --rm \
  -v "${LOGS_DIR:-$(pwd)/logs}:/app/logs" \
  -v "${WORKSPACE_BASE:-$(pwd)/workspaces}:/app/workspaces" \
  -v "$(pwd)/aide/example_tasks:/app/data" \
  -e OPENAI_API_KEY="your-actual-api-key" \
  aide data_dir=/app/data/house_prices goal="Predict price" eval="RMSE"
```

### Development install

```bash
git clone https://github.com/WecoAI/aideml.git
cd aideml && pip install -e .
```

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
