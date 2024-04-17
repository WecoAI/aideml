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

## Solution Gallery

| Domain                           | Task                                                                    | Top%  | Solution Link                                                     | Competition Link                                                                                   |
|:---------------------------------|:------------------------------------------------------------------------|:------|:------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
| Urban Planning                   | Forecast city bikeshare system usage                                    | 5%    | [link](sample_results/bike-sharing-demand.py)                           | [link](https://www.kaggle.com/competitions/bike-sharing-demand/overview)                           |
| Physics                          | Predicting Critical Heat Flux                                           | 56%   | [link](sample_results/playground-series-s3e15.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e15/overview)                       |
| Genomics                         | Classify bacteria species from genomic data                             | 0%    | [link](sample_results/tabular-playground-series-feb-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-feb-2022/overview)            |
| Agriculture                      | Predict blueberry yield                                                 | 58%   | [link](sample_results/playground-series-s3e14.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e14/overview)                       |
| Healthcare                       | Predict disease prognosis                                               | 0%    | [link](sample_results/playground-series-s3e13.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e13/overview)                       |
| Economics                        | Predict monthly microbusiness density in a given area                   | 35%   | [link](sample_results/godaddy-microbusiness-density-forecasting.py)     | [link](https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/overview)     |
| Cryptography                     | Decrypt shakespearean text                                              | 91%   | [link](sample_results/ciphertext-challenge-iii.py)                      | [link](https://www.kaggle.com/competitions/ciphertext-challenge-iii/overview)                      |
| Data Science Education           | Predict passenger survival on Titanic                                   | 78%   | [link](sample_results/tabular-playground-series-apr-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-apr-2021/overview)            |
| Software Engineering             | Predict defects in c programs given various attributes about the code   | 0%    | [link](sample_results/playground-series-s3e23.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e23/overview)                       |
| Real Estate                      | Predict the final price of homes                                        | 5%    | [link](sample_results/home-data-for-ml-course.py)                       | [link](https://www.kaggle.com/competitions/home-data-for-ml-course/overview)                       |
| Real Estate                      | Predict house sale price                                                | 36%   | [link](sample_results/house-prices-advanced-regression-techniques.py)   | [link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)   |
| Entertainment Analytics          | Predict movie worldwide box office revenue                              | 62%   | [link](sample_results/tmdb-box-office-prediction.py)                    | [link](https://www.kaggle.com/competitions/tmdb-box-office-prediction/overview)                    |
| Entertainment Analytics          | Predict scoring probability in next 10 seconds of a rocket league match | 21%   | [link](sample_results/tabular-playground-series-oct-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-oct-2022/overview)            |
| Environmental Science            | Predict air pollution levels                                            | 12%   | [link](sample_results/tabular-playground-series-jul-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jul-2021/overview)            |
| Environmental Science            | Classify forest categories using cartographic variables                 | 55%   | [link](sample_results/forest-cover-type-prediction.py)                  | [link](https://www.kaggle.com/competitions/forest-cover-type-prediction/overview)                  |
| Computer Vision                  | Predict the probability of machine failure                              | 32%   | [link](sample_results/playground-series-s3e17.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e17/overview)                       |
| Computer Vision                  | Identify handwritten digits                                             | 14%   | [link](sample_results/digit-recognizer.py)                              | [link](https://www.kaggle.com/competitions/digit-recognizer/overview)                              |
| Manufacturing                    | Predict missing values in dataset                                       | 70%   | [link](sample_results/tabular-playground-series-jun-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jun-2022/overview)            |
| Manufacturing                    | Predict product failures                                                | 48%   | [link](sample_results/tabular-playground-series-aug-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview)            |
| Manufacturing                    | Cluster control data into different control states                      | 96%   | [link](sample_results/tabular-playground-series-jul-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jul-2022/overview)            |
| Natural Language Processing      | Classify toxic online comments                                          | 78%   | [link](sample_results/jigsaw-toxic-comment-classification-challenge.py) | [link](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) |
| Natural Language Processing      | Predict passenger transport to an alternate dimension                   | 59%   | [link](sample_results/spaceship-titanic.py)                             | [link](https://www.kaggle.com/competitions/spaceship-titanic/overview)                             |
| Natural Language Processing      | Classify sentence sentiment                                             | 42%   | [link](sample_results/sentiment-analysis-on-movie-reviews.py)           | [link](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview)           |
| Natural Language Processing      | Predict whether a tweet is about a real disaster                        | 48%   | [link](sample_results/nlp-getting-started.py)                           | [link](https://www.kaggle.com/competitions/nlp-getting-started/overview)                           |
| Business Analytics               | Predict total sales for each product and store in the next month        | 87%   | [link](sample_results/competitive-data-science-predict-future-sales.py) | [link](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview) |
| Business Analytics               | Predict book sales for 2021                                             | 66%   | [link](sample_results/tabular-playground-series-sep-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/overview)            |
| Business Analytics               | Predict insurance claim amount                                          | 80%   | [link](sample_results/tabular-playground-series-feb-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-feb-2021/overview)            |
| Business Analytics               | Minimize penalty cost in scheduling families to santa's workshop        | 100%  | [link](sample_results/santa-2019-revenge-of-the-accountants.py)         | [link](https://www.kaggle.com/competitions/santa-2019-revenge-of-the-accountants/overview)         |
| Business Analytics               | Predict yearly sales for learning modules                               | 26%   | [link](sample_results/playground-series-s3e19.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e19/overview)                       |
| Business Analytics               | Binary classification of manufacturing machine state                    | 60%   | [link](sample_results/tabular-playground-series-may-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-may-2022/overview)            |
| Business Analytics               | Forecast retail store sales                                             | 36%   | [link](sample_results/tabular-playground-series-jan-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jan-2022/overview)            |
| Business Analytics               | Predict reservation cancellation                                        | 54%   | [link](sample_results/playground-series-s3e7.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e7/overview)                        |
| Finance                          | Predict the probability of an insurance claim                           | 13%   | [link](sample_results/tabular-playground-series-mar-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-mar-2021/overview)            |
| Finance                          | Predict loan loss                                                       | 0%    | [link](sample_results/tabular-playground-series-aug-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-aug-2021/overview)            |
| Finance                          | Predict a continuous target                                             | 42%   | [link](sample_results/tabular-playground-series-jan-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-jan-2021/overview)            |
| Finance                          | Predict customer churn                                                  | 24%   | [link](sample_results/playground-series-s4e1.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s4e1/overview)                        |
| Finance                          | Predict median house value                                              | 58%   | [link](sample_results/playground-series-s3e1.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e1/overview)                        |
| Finance                          | Predict closing price movements for nasdaq listed stocks                | 99%   | [link](sample_results/optiver-trading-at-the-close.py)                  | [link](https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview)                  |
| Finance                          | Predict taxi fare                                                       | 100%  | [link](sample_results/new-york-city-taxi-fare-prediction.py)            | [link](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/overview)            |
| Finance                          | Predict insurance claim probability                                     | 62%   | [link](sample_results/tabular-playground-series-sep-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-sep-2021/overview)            |
| Biotech                          | Predict cat in dat                                                      | 66%   | [link](sample_results/cat-in-the-dat-ii.py)                             | [link](https://www.kaggle.com/competitions/cat-in-the-dat-ii/overview)                             |
| Biotech                          | Predict the biological response of molecules                            | 62%   | [link](sample_results/tabular-playground-series-oct-2021.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-oct-2021/overview)            |
| Biotech                          | Predict medical conditions                                              | 92%   | [link](sample_results/icr-identify-age-related-conditions.py)           | [link](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview)           |
| Biotech                          | Predict wine quality                                                    | 61%   | [link](sample_results/playground-series-s3e5.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e5/overview)                        |
| Biotech                          | Predict binary target without overfitting                               | 98%   | [link](sample_results/dont-overfit-ii.py)                               | [link](https://www.kaggle.com/competitions/dont-overfit-ii/overview)                               |
| Biotech                          | Predict concrete strength                                               | 86%   | [link](sample_results/playground-series-s3e9.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s3e9/overview)                        |
| Biotech                          | Predict crab age                                                        | 46%   | [link](sample_results/playground-series-s3e16.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e16/overview)                       |
| Biotech                          | Predict enzyme characteristics                                          | 10%   | [link](sample_results/playground-series-s3e18.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e18/overview)                       |
| Biotech                          | Classify activity state from sensor data                                | 51%   | [link](sample_results/tabular-playground-series-apr-2022.py)            | [link](https://www.kaggle.com/competitions/tabular-playground-series-apr-2022/overview)            |
| Biotech                          | Predict horse health outcomes                                           | 86%   | [link](sample_results/playground-series-s3e22.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e22/overview)                       |
| Biotech                          | Predict the mohs hardness of a mineral                                  | 64%   | [link](sample_results/playground-series-s3e25.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e25/overview)                       |
| Biotech                          | Predict cirrhosis patient outcomes                                      | 51%   | [link](sample_results/playground-series-s3e26.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e26/overview)                       |
| Biotech                          | Predict obesity risk                                                    | 62%   | [link](sample_results/playground-series-s4e2.py)                        | [link](https://www.kaggle.com/competitions/playground-series-s4e2/overview)                        |
| Biotech                          | Classify presence of feature in data                                    | 66%   | [link](sample_results/cat-in-the-dat.py)                                | [link](https://www.kaggle.com/competitions/cat-in-the-dat/overview)                                |
| Biotech                          | Predict patient's smoking status                                        | 40%   | [link](sample_results/playground-series-s3e24.py)                       | [link](https://www.kaggle.com/competitions/playground-series-s3e24/overview)                       |
