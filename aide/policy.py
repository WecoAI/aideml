import logging
import queue
import random
from dataclasses import dataclass
from typing import Literal, Protocol

from .backend import FunctionSpec, query
from .controller_trace import log_controller_event
from .journal import Journal, Node
from .rlhf.observation import build_observation, task_desc_to_string
from .utils.config import SearchConfig

logger = logging.getLogger("aide")


@dataclass
class SearchAction:
    kind: Literal["draft", "debug", "improve"]
    parent_id: str | None = None
    rationale: str | None = None
    hint: str | None = None
    is_invalid: bool = False


class SearchPolicy(Protocol):
    def select(
        self,
        journal: Journal,
        task_desc: str,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
    ) -> SearchAction: ...


def _node_by_id(journal: Journal, node_id: str | None) -> Node | None:
    if node_id is None:
        return None
    for node in journal.nodes:
        if node.id == node_id:
            return node
    return None


def validate_action(
    action: SearchAction, journal: Journal, search_cfg: SearchConfig
) -> tuple[bool, str]:
    if action.kind == "draft":
        if action.parent_id is not None:
            return False, "draft must not include parent_id"
        return True, ""

    parent = _node_by_id(journal, action.parent_id)
    if parent is None:
        return False, "parent_id not found"

    if action.kind == "debug":
        if not parent.is_buggy:
            return False, "debug requires buggy parent"
        if not parent.is_leaf:
            return False, "debug requires leaf parent"
        if parent.debug_depth > search_cfg.max_debug_depth:
            return False, "debug depth exceeds max_debug_depth"
        return True, ""

    if action.kind == "improve":
        if parent.is_buggy:
            return False, "improve requires non-buggy parent"
        return True, ""

    return False, f"unknown action kind: {action.kind}"


class HeuristicPolicy:
    def select(
        self,
        journal: Journal,
        task_desc: str,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
    ) -> SearchAction:
        del task_desc, step_idx, total_steps

        if len(journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return SearchAction(kind="draft")

        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                n
                for n in journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                node = random.choice(debuggable_nodes)
                logger.debug("[search policy] debugging")
                return SearchAction(kind="debug", parent_id=node.id)
            logger.debug("[search policy] not debugging by chance")

        good_nodes = journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return SearchAction(kind="draft")

        greedy_node = journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return SearchAction(kind="improve", parent_id=greedy_node.id)  # type: ignore[arg-type]


select_action_func_spec = FunctionSpec(
    name="select_action",
    json_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["draft", "debug", "improve"]},
            "parent_id": {"type": ["string", "null"]},
            "rationale": {"type": "string"},
        },
        "required": ["action", "parent_id", "rationale"],
    },
    description="Select the next tree-search action.",
)


class LLMPolicy:
    def __init__(self):
        self.fallback = HeuristicPolicy()
        self.invalid_action_count = 0

    def _build_observation(
        self,
        journal: Journal,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
        task_desc: str,
    ) -> str:
        maximize = True
        for n in journal.nodes:
            if n.metric is not None and n.metric.maximize is not None:
                maximize = bool(n.metric.maximize)
                break
        best_node = journal.get_best_node(only_good=True)
        best_so_far = (
            float(best_node.metric.value)
            if best_node and best_node.metric and best_node.metric.value is not None
            else None
        )
        baseline = 1.0
        summaries = journal.summary_for_policy(max_nodes=search_cfg.policy_max_obs_nodes)
        return build_observation(
            summaries,
            task_desc_to_string(task_desc),
            baseline_metric=baseline,
            maximize=maximize,
            step_idx=step_idx,
            total_steps=total_steps,
            best_metric_so_far=best_so_far,
            num_drafts=len(journal.draft_nodes),
            n_debuggable_leaves=len(
                [
                    n
                    for n in journal.buggy_nodes
                    if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
                ]
            ),
            n_good_nodes=len(journal.good_nodes),
            recent_actions=[],
        )

    def select(
        self,
        journal: Journal,
        task_desc: str,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
    ) -> SearchAction:
        if not search_cfg.policy_model:
            return self.fallback.select(journal, task_desc, search_cfg, step_idx, total_steps)

        obs_text = self._build_observation(
            journal=journal,
            search_cfg=search_cfg,
            step_idx=step_idx,
            total_steps=total_steps,
            task_desc=task_desc,
        )
        prompt = {
            "Task": task_desc,
            "Observation": obs_text,
        }
        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=select_action_func_spec,
            model=search_cfg.policy_model,
            temperature=search_cfg.policy_temp,
        )
        assert isinstance(response, dict)
        action = SearchAction(
            kind=response["action"],
            parent_id=response["parent_id"],
            rationale=response["rationale"],
        )
        is_valid, reason = validate_action(action, journal, search_cfg)
        if is_valid:
            return action

        self.invalid_action_count += 1
        logger.warning("Invalid LLM policy action: %s", reason)
        fb = self.fallback.select(journal, task_desc, search_cfg, step_idx, total_steps)
        fb.is_invalid = True
        return fb


class ControllerPolicy:
    """Learned controller policy: decides action + hint per candidate node."""

    def __init__(self):
        from .controller import LLMController

        self.controller = LLMController()
        self.fallback = HeuristicPolicy()
        self.invalid_action_count = 0

    def _candidate_nodes(
        self, journal: Journal, search_cfg: SearchConfig
    ) -> list[Node]:
        candidates: list[Node] = []
        best = journal.get_best_node(only_good=True)
        if best is not None:
            candidates.append(best)

        debuggable = [
            n
            for n in journal.buggy_nodes
            if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
        ]
        for n in debuggable:
            if n not in candidates:
                candidates.append(n)
        return candidates

    def select(
        self,
        journal: Journal,
        task_desc: str,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
    ) -> SearchAction:
        if len(journal.draft_nodes) < search_cfg.num_drafts:
            log_controller_event(
                "draft_phase",
                step_idx=step_idx,
                n_drafts=len(journal.draft_nodes),
                num_drafts=search_cfg.num_drafts,
            )
            return SearchAction(kind="draft")

        candidates = self._candidate_nodes(journal, search_cfg)
        abandoned: set[str] = set()

        for candidate in candidates:
            if candidate.id in abandoned:
                continue
            out = self.controller.decide(
                candidate, task_desc, journal, search_cfg
            )
            if out is None:
                continue

            if out.action == "abandon":
                abandoned.add(candidate.id)
                continue

            if out.action == "debug":
                action = SearchAction(
                    kind="debug",
                    parent_id=candidate.id,
                    rationale="controller",
                    hint=out.hint,
                )
            elif out.action == "improve":
                action = SearchAction(
                    kind="improve",
                    parent_id=candidate.id,
                    rationale="controller",
                    hint=out.hint,
                )
            else:
                abandoned.add(candidate.id)
                continue

            is_valid, reason = validate_action(action, journal, search_cfg)
            if is_valid:
                return action

            self.invalid_action_count += 1
            logger.warning("Invalid controller action on %s: %s", candidate.id, reason)
            abandoned.add(candidate.id)

        fb = self.fallback.select(
            journal, task_desc, search_cfg, step_idx=len(journal), total_steps=0
        )
        fb.is_invalid = True
        return fb


class UnifiedControllerPolicy:
    """Tree selection driven by the unified analyzer's per-node suggestions.

    The trained analyzer reviews each node right after execution and caches
    (next_action, next_hint, next_confidence) on it, so selection here needs no
    extra LLM calls: rank actionable nodes by the analyzer's confidence.
    """

    def __init__(self):
        self.fallback = HeuristicPolicy()

    @staticmethod
    def _metric_or_none(node: Node) -> float | None:
        if node.metric is None or node.metric.value is None:
            return None
        return float(node.metric.value)

    def select(
        self,
        journal: Journal,
        task_desc: str,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
    ) -> SearchAction:
        if len(journal.draft_nodes) < search_cfg.num_drafts:
            log_controller_event(
                "draft_phase",
                step_idx=step_idx,
                n_drafts=len(journal.draft_nodes),
                num_drafts=search_cfg.num_drafts,
            )
            return SearchAction(kind="draft")

        candidates: list[tuple[float, float, SearchAction]] = []
        n_abandoned = 0

        for node in journal.buggy_nodes:
            if not (node.is_leaf and node.debug_depth <= search_cfg.max_debug_depth):
                continue
            if node.next_action == "abandon":
                n_abandoned += 1
            elif node.next_action == "debug":
                conf = node.next_confidence if node.next_confidence is not None else 0.5
                candidates.append(
                    (
                        conf,
                        float("-inf"),
                        SearchAction(
                            kind="debug",
                            parent_id=node.id,
                            rationale="unified_analyzer",
                            hint=node.next_hint,
                        ),
                    )
                )

        for node in journal.good_nodes:
            if node.next_action == "abandon":
                n_abandoned += 1
            elif node.next_action == "improve":
                conf = node.next_confidence if node.next_confidence is not None else 0.5
                metric = self._metric_or_none(node)
                candidates.append(
                    (
                        conf,
                        metric if metric is not None else float("-inf"),
                        SearchAction(
                            kind="improve",
                            parent_id=node.id,
                            rationale="unified_analyzer",
                            hint=node.next_hint,
                        ),
                    )
                )

        for conf, metric, action in sorted(
            candidates, key=lambda c: (c[0], c[1]), reverse=True
        ):
            is_valid, reason = validate_action(action, journal, search_cfg)
            if is_valid:
                log_controller_event(
                    "unified_select",
                    step_idx=step_idx,
                    action=action.kind,
                    parent_id=action.parent_id,
                    confidence=conf,
                )
                return action
            logger.warning(
                "Invalid unified suggestion on %s: %s", action.parent_id, reason
            )

        if n_abandoned > 0:
            # The analyzer abandoned every actionable branch: honor it by
            # drafting a fresh solution instead of grinding on rejected nodes.
            log_controller_event(
                "unified_all_abandoned", step_idx=step_idx, n_abandoned=n_abandoned
            )
            return SearchAction(kind="draft", rationale="unified_analyzer_abandon")

        # No suggestions available (e.g. nodes reviewed by the GPT fallback);
        # fall back to the heuristic.
        return self.fallback.select(journal, task_desc, search_cfg, step_idx, total_steps)


class HeuristicPlusControllerPolicy:
    """Heuristic search policy with controller hints on the chosen parent."""

    def __init__(self):
        from .controller import build_controller

        self.heuristic = HeuristicPolicy()
        self._controller = None
        self._search_cfg: SearchConfig | None = None

    def _get_controller(self, search_cfg: SearchConfig):
        if self._search_cfg is not search_cfg:
            from .controller import build_controller

            self._controller = build_controller(search_cfg)
            self._search_cfg = search_cfg
        return self._controller

    def select(
        self,
        journal: Journal,
        task_desc: str,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
    ) -> SearchAction:
        action = self.heuristic.select(
            journal, task_desc, search_cfg, step_idx, total_steps
        )
        if action.parent_id is None:
            return action

        parent = _node_by_id(journal, action.parent_id)
        if parent is None:
            return action

        controller = self._get_controller(search_cfg)
        if controller is None:
            return action

        out = controller.decide(parent, task_desc, journal, search_cfg)
        if out is None:
            return action
        if out.action == "abandon":
            # Abandon hints are trained as "drop this branch" guidance; injecting
            # them into an improve/debug prompt would mislead the coding model.
            log_controller_event(
                "hint_dropped_abandon",
                node_id=parent.id,
                heuristic_action=action.kind,
            )
            return action
        if out.hint:
            action.hint = out.hint
        return action


class ExternalPolicy:
    def __init__(self):
        self._queue: queue.Queue[SearchAction] = queue.Queue()

    def put_action(self, action: SearchAction) -> None:
        self._queue.put(action)

    def select(
        self,
        journal: Journal,
        task_desc: str,
        search_cfg: SearchConfig,
        step_idx: int,
        total_steps: int,
    ) -> SearchAction:
        del journal, task_desc, search_cfg, step_idx, total_steps
        return self._queue.get_nowait()
