"""Parser and test harness for degenerate Nash equilibrium cases.

Parses log output from failed Nash computations and provides tools to
reproduce and test fixes.

Workflow:
    1. Parse raw logs: cases = parse_raw_log("no_nash_eq.txt")
    2. Save to JSON:   save_cases(cases, "degenerate_cases.json")
    3. Load later:     cases = load_cases("degenerate_cases.json")
    4. Test fixes:     results = test_nash_on_cases(cases)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alpharat.mcts.nash import compute_nash_equilibrium


@dataclass
class DegenerateCase:
    """A single case where Nash equilibrium computation failed or was degenerate."""

    p1_payoffs: np.ndarray  # [5, 5]
    p2_payoffs: np.ndarray  # [5, 5]
    p1_prior: np.ndarray  # [5]
    p2_prior: np.ndarray  # [5]
    p1_effective: list[int]  # [5] mapping
    p2_effective: list[int]  # [5] mapping
    action_visits: np.ndarray  # [5, 5]

    @property
    def payout_matrix(self) -> np.ndarray:
        """Combined payout matrix [2, 5, 5]."""
        return np.stack([self.p1_payoffs, self.p2_payoffs], axis=0)

    def equivalent_rows_differ(self, atol: float = 1e-6) -> dict[str, list[tuple[int, int]]]:
        """Check if equivalent actions have different payout values.

        Returns dict with 'p1' and 'p2' keys, each containing list of
        (action_a, action_b) pairs that should be equivalent but differ.
        """
        result: dict[str, list[tuple[int, int]]] = {"p1": [], "p2": []}

        # Check P1's rows (actions with same effective should have same row)
        for i in range(5):
            for j in range(i + 1, 5):
                if self.p1_effective[i] == self.p1_effective[j] and (
                    not np.allclose(self.p1_payoffs[i], self.p1_payoffs[j], atol=atol)
                    or not np.allclose(self.p2_payoffs[i], self.p2_payoffs[j], atol=atol)
                ):
                    result["p1"].append((i, j))

        # Check P2's columns
        for i in range(5):
            for j in range(i + 1, 5):
                if self.p2_effective[i] == self.p2_effective[j] and (
                    not np.allclose(self.p1_payoffs[:, i], self.p1_payoffs[:, j], atol=atol)
                    or not np.allclose(self.p2_payoffs[:, i], self.p2_payoffs[:, j], atol=atol)
                ):
                    result["p2"].append((i, j))

        return result

    def has_equivalence_violation(self, atol: float = 1e-6) -> bool:
        """True if any equivalent actions have different payout values."""
        violations = self.equivalent_rows_differ(atol)
        return bool(violations["p1"]) or bool(violations["p2"])

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "p1_payoffs": self.p1_payoffs.tolist(),
            "p2_payoffs": self.p2_payoffs.tolist(),
            "p1_prior": self.p1_prior.tolist(),
            "p2_prior": self.p2_prior.tolist(),
            "p1_effective": self.p1_effective,
            "p2_effective": self.p2_effective,
            "action_visits": self.action_visits.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DegenerateCase:
        """Create from dict (loaded from JSON)."""
        return cls(
            p1_payoffs=np.array(d["p1_payoffs"]),
            p2_payoffs=np.array(d["p2_payoffs"]),
            p1_prior=np.array(d["p1_prior"]),
            p2_prior=np.array(d["p2_prior"]),
            p1_effective=d["p1_effective"],
            p2_effective=d["p2_effective"],
            action_visits=np.array(d["action_visits"]),
        )


# =============================================================================
# Persistence (JSON)
# =============================================================================


def save_cases(cases: list[DegenerateCase], filepath: str | Path) -> None:
    """Save cases to JSON file."""
    filepath = Path(filepath)
    data = [case.to_dict() for case in cases]
    filepath.write_text(json.dumps(data, indent=2))


def load_cases(filepath: str | Path) -> list[DegenerateCase]:
    """Load cases from JSON file."""
    filepath = Path(filepath)
    data = json.loads(filepath.read_text())
    return [DegenerateCase.from_dict(d) for d in data]


# =============================================================================
# Raw log parsing
# =============================================================================


def _parse_matrix(lines: list[str], start_idx: int, rows: int = 5) -> tuple[np.ndarray, int]:
    """Parse a numpy-style matrix from log lines."""
    matrix_lines: list[list[float]] = []
    idx = start_idx

    while idx < len(lines) and len(matrix_lines) < rows:
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # Remove line number prefix (e.g., "     3→")
        if "→" in line:
            line = line.split("→", 1)[1]

        # Handle numpy multi-line arrays
        line = line.replace("[[", "[").replace("]]", "]")
        line = line.strip("[] ")

        if line:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if nums:
                matrix_lines.append([float(n) for n in nums])

        idx += 1

    return np.array(matrix_lines), idx


def _parse_array(line: str) -> np.ndarray:
    """Parse a 1D numpy array from a log line."""
    if ":" in line:
        line = line.split(":", 1)[1]
    line = line.strip("[] ")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    return np.array([float(n) for n in nums])


def _parse_int_list(line: str) -> list[int]:
    """Parse a list of ints like '[0, 1, 2, 4, 4]'."""
    if ":" in line:
        line = line.split(":", 1)[1]
    nums = re.findall(r"\d+", line)
    return [int(n) for n in nums]


def _parse_single_case(lines: list[str]) -> DegenerateCase | None:
    """Parse a single case from its lines."""
    p1_payoffs = None
    p2_payoffs = None
    p1_prior = None
    p2_prior = None
    p1_effective = None
    p2_effective = None
    action_visits = None

    i = 0
    while i < len(lines):
        line = lines[i]

        if "→" in line:
            line = line.split("→", 1)[1]

        line_lower = line.lower().strip()

        if "p1 payoffs:" in line_lower:
            p1_payoffs, i = _parse_matrix(lines, i + 1)
        elif "p2 payoffs:" in line_lower:
            p2_payoffs, i = _parse_matrix(lines, i + 1)
        elif "p1 prior:" in line_lower:
            p1_prior = _parse_array(line)
            i += 1
        elif "p2 prior:" in line_lower:
            p2_prior = _parse_array(line)
            i += 1
        elif "p1 effective:" in line_lower:
            p1_effective = _parse_int_list(line)
            i += 1
        elif "p2 effective:" in line_lower:
            p2_effective = _parse_int_list(line)
            i += 1
        elif "action visits:" in line_lower:
            action_visits, i = _parse_matrix(lines, i + 1)
        else:
            i += 1

    # Validate we got everything
    if (
        p1_payoffs is None
        or p2_payoffs is None
        or p1_prior is None
        or p2_prior is None
        or p1_effective is None
        or p2_effective is None
    ):
        return None

    if action_visits is None:
        action_visits = np.ones((5, 5))

    return DegenerateCase(
        p1_payoffs=p1_payoffs,
        p2_payoffs=p2_payoffs,
        p1_prior=p1_prior,
        p2_prior=p2_prior,
        p1_effective=p1_effective,
        p2_effective=p2_effective,
        action_visits=action_visits,
    )


def parse_raw_log(filepath: str | Path) -> list[DegenerateCase]:
    """Parse degenerate cases from raw log file.

    Looks for "No Nash equilibrium found" markers and extracts the
    payout matrices, priors, effective mappings, and visit counts.

    Args:
        filepath: Path to the raw log file (e.g., no_nash_eq.txt).

    Returns:
        List of parsed DegenerateCase objects.
    """
    filepath = Path(filepath)
    content = filepath.read_text()

    # Split on the marker
    case_pattern = r"(?:WARNING - )?No Nash equilibrium found"
    parts = re.split(case_pattern, content)

    cases = []
    for part in parts[1:]:  # Skip first empty part
        lines = part.strip().split("\n")
        if not lines:
            continue
        try:
            case = _parse_single_case(lines)
            if case is not None:
                cases.append(case)
        except (ValueError, IndexError):
            continue

    return cases


# =============================================================================
# Testing
# =============================================================================


@dataclass
class NashTestResult:
    """Result of testing Nash computation on a degenerate case."""

    case_idx: int
    success: bool  # Did we find a non-uniform equilibrium?
    fell_back_to_uniform: bool
    has_equivalence_violation: bool
    p1_strategy: np.ndarray | None = None
    p2_strategy: np.ndarray | None = None
    error: str | None = None


def _is_uniform_on_effective(strategy: np.ndarray, effective_map: list[int]) -> bool:
    """Check if strategy is uniform over effective actions."""
    effective_actions = sorted(set(effective_map))
    n_effective = len(effective_actions)
    if n_effective == 0:
        return True

    expected_prob = 1.0 / n_effective
    for action in effective_actions:
        if not np.isclose(strategy[action], expected_prob, atol=1e-4):
            return False
    return True


def test_nash_on_cases(
    cases: list[DegenerateCase],
    min_visits: int = 5,
) -> list[NashTestResult]:
    """Test current Nash computation on a list of cases.

    Args:
        cases: List of degenerate cases to test.
        min_visits: Minimum visits threshold for filtering.

    Returns:
        List of test results.
    """
    results = []

    for idx, case in enumerate(cases):
        try:
            p1_strat, p2_strat = compute_nash_equilibrium(
                case.payout_matrix,
                p1_effective=case.p1_effective,
                p2_effective=case.p2_effective,
                prior_p1=case.p1_prior,
                prior_p2=case.p2_prior,
                action_visits=case.action_visits,
                min_visits=min_visits,
            )

            is_uniform = _is_uniform_on_effective(
                p1_strat, case.p1_effective
            ) and _is_uniform_on_effective(p2_strat, case.p2_effective)

            results.append(
                NashTestResult(
                    case_idx=idx,
                    success=not is_uniform,
                    fell_back_to_uniform=is_uniform,
                    has_equivalence_violation=case.has_equivalence_violation(),
                    p1_strategy=p1_strat,
                    p2_strategy=p2_strat,
                )
            )
        except Exception as e:
            results.append(
                NashTestResult(
                    case_idx=idx,
                    success=False,
                    fell_back_to_uniform=False,
                    has_equivalence_violation=case.has_equivalence_violation(),
                    error=str(e),
                )
            )

    return results


def summarize_results(results: list[NashTestResult]) -> dict:
    """Summarize test results."""
    n_total = len(results)
    n_success = sum(1 for r in results if r.success)
    n_uniform = sum(1 for r in results if r.fell_back_to_uniform)
    n_errors = sum(1 for r in results if r.error is not None)
    n_violations = sum(1 for r in results if r.has_equivalence_violation)

    return {
        "total": n_total,
        "found_equilibrium": n_success,
        "fell_back_to_uniform": n_uniform,
        "errors": n_errors,
        "with_equivalence_violations": n_violations,
        "success_rate": n_success / n_total if n_total > 0 else 0.0,
    }
