import logging
import random
from typing import Any, Callable, cast

import humanize
from .backend import FunctionSpec, query_with_usage
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .policy import HeuristicPolicy, SearchAction, SearchPolicy
from .rlhf.observation import task_desc_to_string
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        policy: SearchPolicy | None = None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.policy = policy or HeuristicPolicy()
        self._unified_analyzer = None
        if (
            self.acfg.search.controller_kind == "unified"
            and self.acfg.search.controller_model
        ):
            from .controller import UnifiedAnalyzer

            self._unified_analyzer = UnifiedAnalyzer()

    def search_policy(self) -> Node | None:
        """Backward-compatible wrapper around heuristic selection."""
        action = HeuristicPolicy().select(
            journal=self.journal,
            task_desc=task_desc_to_string(self.task_desc),
            search_cfg=self.acfg.search,
            step_idx=len(self.journal),
            total_steps=self.acfg.steps,
        )
        if action.kind == "draft":
            return None
        return next((n for n in self.journal.nodes if n.id == action.parent_id), None)

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str, dict]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        code_usage: dict = {}
        for _ in range(retries):
            completion_text, code_usage = query_with_usage(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                call_type="code",
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code, code_usage

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text or "", code_usage  # type: ignore

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code, code_usage = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, token_usage={"code": code_usage})

    def _controller_hint_block(self, hint: str | None) -> dict[str, str] | None:
        if not hint:
            return None
        return {
            "Controller hint": (
                f"{hint}\n\n"
                "Use this hint as strategic guidance. You may ignore it if it conflicts "
                "with the execution output or dataset schema."
            )
        }

    def _improve(self, parent_node: Node, hint: str | None = None) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }
        hint_block = self._controller_hint_block(hint)
        if hint_block:
            prompt |= hint_block

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code, code_usage = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            hint=hint,
            token_usage={"code": code_usage},
        )

    def _debug(self, parent_node: Node, hint: str | None = None) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        hint_block = self._controller_hint_block(hint)
        if hint_block:
            prompt |= hint_block
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code, code_usage = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            hint=hint,
            token_usage={"code": code_usage},
        )

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        # JSON string form matches the controller's training prompts (task_desc_to_string
        # was also used at export time); str(dict) would produce a Python repr instead.
        action: SearchAction = self.policy.select(
            journal=self.journal,
            task_desc=task_desc_to_string(self.task_desc),
            search_cfg=self.acfg.search,
            step_idx=len(self.journal),
            total_steps=self.acfg.steps,
        )
        parent_node = (
            next((n for n in self.journal.nodes if n.id == action.parent_id), None)
            if action.parent_id is not None
            else None
        )
        logger.debug("Agent is generating code, action=%s", action.kind)

        hint = action.hint

        if action.kind == "draft" or parent_node is None:
            result_node = self._draft()
        elif action.kind == "debug":
            result_node = self._debug(parent_node, hint=hint)
        else:
            result_node = self._improve(parent_node, hint=hint)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        if self._unified_analyzer is not None:
            if self._parse_exec_result_unified(node):
                return
            logger.warning(
                "Unified analyzer failed for node %s; falling back to GPT review", node.id
            )

        self._parse_exec_result_gpt(node)

    def _parse_exec_result_unified(self, node: Node) -> bool:
        """Review + next-step suggestion in one trained-model call. Returns False on failure."""
        out = self._unified_analyzer.analyze(node, self.task_desc, self.acfg.search)  # type: ignore[union-attr]
        if out is None:
            return False

        node.analysis = out.analysis
        node.is_buggy = out.is_bug or node.exc_type is not None or out.metric is None
        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(out.metric, maximize=not out.lower_is_better)

        node.next_action = out.action
        node.next_hint = out.hint or None
        node.next_confidence = out.confidence
        return True

    def _parse_exec_result_gpt(self, node: Node):
        prompt = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            ),
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        response, review_usage = query_with_usage(
            system_message=prompt,
            user_message=None,
            func_spec=review_func_spec,
            model=self.acfg.feedback.model,
            temperature=self.acfg.feedback.temp,
            call_type="review",
        )
        response = cast(dict, response)

        if node.token_usage is None:
            node.token_usage = {}
        node.token_usage["review"] = review_usage

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
