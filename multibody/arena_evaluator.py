"""Score candidates: every candidate plays one match on each of the run's layouts (R14, R15).

- ``"reference"``: the candidate is the Learner in slot ``layout mod K``, the
  other K-1 robots run the Reference_Layer; the candidate's match score is the
  Learner's F (R14.1-14.4).
- ``"self"``: all K robots run the candidate; its match score is the mean of the
  K F values (R14.5).
- A candidate's score is the mean of its ``NUM_LAYOUTS`` match scores, in layout
  order (R15.1).

Matches run in up to ``worker_count`` worker processes, at most two per worker
in flight; results are put back by (candidate, layout), so the scores do not
depend on which worker finishes first, and ``worker_count == 1`` runs inline
with the same result (R15.3, R15.5, R15.6).

Failures (R15.7): a deterministic failure (the scorer's ``MatchFailure``) gives
every robot in that match FAILED_MATCH_SCORE and the generation continues; a
worker process dying stops the generation (:class:`MatchWorkerCrashed`).
"""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any

import numpy as np

import contest_paths

contest_paths.install()

import match  # noqa: E402
import match_worker  # noqa: E402
from arena import Layout  # noqa: E402
from config_loader import ContestSettings  # noqa: E402
from match_scoring import RobotMetrics  # noqa: E402


class MatchWorkerCrashed(Exception):
    """A worker process died before returning a match's result (R15.7b)."""

    def __init__(self, candidate_index: int, layout_index: int, detail: str) -> None:
        super().__init__(
            f"the worker running candidate {candidate_index}, layout {layout_index} "
            f"died before returning its result ({detail})"
        )
        self.candidate_index = candidate_index
        self.layout_index = layout_index


@dataclass
class MatchRecord:
    """One match of one candidate: who played which role, and how it went."""

    candidate_index: int
    layout_index: int
    roles: list[str]
    metrics: list[RobotMetrics] | None  # None when the match failed
    failure: str | None
    match_score: float  # the candidate's score for this match


@dataclass
class CandidateRecords:
    """All matches of one generation's evaluation."""

    matches: list[MatchRecord]  # ordered by (candidate, layout)

    @property
    def failed_count(self) -> int:
        """Number of matches that failed the deterministic checks."""
        return sum(1 for record in self.matches if record.failure is not None)

    @property
    def failed_fraction(self) -> float:
        """Fraction of matches that failed the deterministic checks."""
        return self.failed_count / len(self.matches) if self.matches else 0.0


class MatchEvaluator:
    """Candidates x layouts -> candidate scores and per-match records."""

    def __init__(
        self,
        settings: ContestSettings,
        cpg_network_structure: Any,
        output_mapping: list[Any],
        layouts: list[Layout],
        per_robot_body_copy: bool = False,
    ) -> None:
        """
        :param settings: The contest settings.
        :param cpg_network_structure: From ``config_loader.body_wiring``.
        :param output_mapping: From ``config_loader.body_wiring``.
        :param layouts: The run's layouts.
        :param per_robot_body_copy: The R21.7 fallback switch.
        """
        self._settings = settings
        self._structure = cpg_network_structure
        self._mapping = output_mapping
        self._layouts = layouts
        self._per_robot_body_copy = per_robot_body_copy

    def make_task(self, candidate: np.ndarray, candidate_index: int, layout_index: int, reference_layer: np.ndarray) -> dict[str, Any]:
        """
        The match task for one (candidate, layout).

        :returns: The task.
        """
        controllers = match.slot_controllers_for(self._settings, candidate, layout_index, reference_layer)
        return match.make_match_task(
            self._settings,
            self._layouts[layout_index],
            layout_index,
            controllers,
            self._structure,
            self._mapping,
            scene_id=candidate_index * len(self._layouts) + layout_index,
            per_robot_body_copy=self._per_robot_body_copy,
        )

    def score_candidates(
        self, candidate_layers: list[np.ndarray], reference_layer: np.ndarray
    ) -> tuple[list[float], CandidateRecords]:
        """
        Score every candidate on every layout.

        :param candidate_layers: The candidates' Layer_Parameters.
        :param reference_layer: The Opponents' Layer_Parameters (``"reference"`` mode).
        :returns: One score per candidate, and the match records.
        :raises MatchWorkerCrashed: If a worker process dies.
        """
        num_layouts = len(self._layouts)
        total = len(candidate_layers) * num_layouts
        outcomes: list[match_worker.MatchOutcome | None] = [None] * total
        roles: list[list[str] | None] = [None] * total

        def task_for(index: int) -> dict[str, Any]:
            candidate_index, layout_index = divmod(index, num_layouts)
            task = self.make_task(
                np.asarray(candidate_layers[candidate_index], dtype=np.float64),
                candidate_index,
                layout_index,
                reference_layer,
            )
            roles[index] = task["roles"]
            return task

        workers = max(1, int(self._settings.worker_count))
        if workers == 1:
            for index in range(total):
                try:
                    outcomes[index] = match_worker.simulate_and_score_match(task_for(index))
                except Exception as error:  # noqa: BLE001 - same reporting as the pool path
                    candidate_index, layout_index = divmod(index, num_layouts)
                    raise MatchWorkerCrashed(candidate_index, layout_index, f"{type(error).__name__}: {error}") from error
        else:
            self._run_in_pool(task_for, outcomes, total, workers, num_layouts)

        records = []
        scores = []
        for candidate_index in range(len(candidate_layers)):
            match_scores = []
            for layout_index in range(num_layouts):
                index = candidate_index * num_layouts + layout_index
                outcome = outcomes[index]
                match_score = self._match_score(outcome, roles[index])
                match_scores.append(match_score)
                if outcome.failure is not None:
                    logging.warning(
                        f"Match failed (scored {self._settings.failed_match_score}): candidate "
                        f"{candidate_index}, layout {layout_index}: {outcome.failure}"
                    )
                records.append(
                    MatchRecord(
                        candidate_index=candidate_index,
                        layout_index=layout_index,
                        roles=list(roles[index]),
                        metrics=outcome.metrics,
                        failure=outcome.failure,
                        match_score=match_score,
                    )
                )
            scores.append(float(sum(match_scores) / num_layouts))
        return scores, CandidateRecords(matches=records)

    def _match_score(self, outcome: match_worker.MatchOutcome, roles: list[str]) -> float:
        """The candidate's score for one match (R14.2, R14.5, D-13)."""
        if outcome.failure is not None:
            return float(self._settings.failed_match_score)
        if self._settings.opponents == "self":
            return float(sum(m.score for m in outcome.metrics) / len(outcome.metrics))
        return float(outcome.metrics[roles.index("Learner")].score)

    @staticmethod
    def _run_in_pool(task_for: Any, outcomes: list[Any], total: int, workers: int, num_layouts: int) -> None:
        """Run every task in a worker pool, putting results back by index."""
        executor = ProcessPoolExecutor(max_workers=workers)
        pending: dict[Any, int] = {}
        next_index = done = 0
        report_every = max(1, total // 10)
        try:
            while done < total:
                while next_index < total and len(pending) < 2 * workers:
                    future = executor.submit(match_worker.simulate_and_score_match, task_for(next_index))
                    pending[future] = next_index
                    next_index += 1
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    try:
                        outcomes[index] = future.result()
                    except BrokenProcessPool as error:
                        candidate_index, layout_index = divmod(index, num_layouts)
                        raise MatchWorkerCrashed(candidate_index, layout_index, str(error) or "process pool broken") from error
                    except Exception as error:  # noqa: BLE001 - any other error also stops the generation
                        candidate_index, layout_index = divmod(index, num_layouts)
                        raise MatchWorkerCrashed(candidate_index, layout_index, f"{type(error).__name__}: {error}") from error
                    done += 1
                    if done % report_every == 0 or done == total:
                        logging.info(f"Finished {done} / {total} matches.")
        except BaseException:
            # Stop everything now instead of letting in-flight matches run on.
            processes = list(getattr(executor, "_processes", {}).values())
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                process.terminate()
            raise
        executor.shutdown(wait=True)
