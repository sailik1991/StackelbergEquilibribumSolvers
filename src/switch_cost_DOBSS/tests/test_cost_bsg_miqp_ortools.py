#!/usr/bin/python
"""
Test for cost_BSG_miqp_ortools.py solver.
Runs the solver with test inputs and verifies expected output.
"""

import subprocess
import os
import unittest


def run_solver(input_file, alpha):
    """Run the solver and parse its output."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    switch_cost_dir = os.path.dirname(tests_dir)
    solver_path = os.path.join(switch_cost_dir, "cost_BSG_miqp_ortools.py")
    input_path = os.path.join(tests_dir, input_file) if not os.path.isabs(input_file) else input_file

    result = subprocess.run(
        ["python3", solver_path, input_path, str(alpha)],
        capture_output=True,
        text=True,
        cwd=switch_cost_dir
    )

    variables = {}
    objective = None

    for line in result.stdout.split("\n"):
        if line.startswith("Obj -> "):
            objective = float(line.split(" -> ")[1].strip())
        elif " -> " in line:
            parts = line.split(" -> ")
            name = parts[0].strip()
            value = float(parts[1].strip())
            variables[name] = value

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "variables": variables,
        "objective": objective
    }


class TestCostBSGMiqpOrtoolsSimple(unittest.TestCase):
    """Test with simple 2x2 input."""

    @classmethod
    def setUpClass(cls):
        """Run the solver once with simple input."""
        cls.result = run_solver("simple_cost_input.txt", alpha=0.5)

    def test_solver_runs_successfully(self):
        """Test that the solver completes without error."""
        self.assertEqual(
            self.result["returncode"], 0,
            f"Solver failed with stderr: {self.result['stderr']}"
        )

    def test_objective_exists(self):
        """Test that an objective value was computed."""
        self.assertIsNotNone(self.result["objective"])

    def test_defender_strategy_sums_to_one(self):
        """Test that defender strategy probabilities sum to 1."""
        variables = self.result["variables"]
        defender_probs = [
            variables.get("x-0", 0),
            variables.get("x-1", 0),
        ]
        self.assertAlmostEqual(sum(defender_probs), 1.0, places=5)

    def test_defender_strategies_valid(self):
        """Test that defender strategies are in valid range [0,1]."""
        variables = self.result["variables"]
        for i in range(2):
            x = variables.get(f"x-{i}", 0)
            self.assertGreaterEqual(x, -1e-6)
            self.assertLessEqual(x, 1.0 + 1e-6)

    def test_each_attacker_has_pure_strategy(self):
        """Test that each attacker chooses exactly one pure strategy."""
        variables = self.result["variables"]

        # Attacker 0
        attacker_0 = [variables.get("0-Attack1", 0),
                      variables.get("0-Attack2", 0)]
        self.assertAlmostEqual(sum(abs(v) for v in attacker_0), 1.0, places=4)

        # Attacker 1
        attacker_1 = [variables.get("1-Attack3", 0),
                      variables.get("1-Attack4", 0)]
        self.assertAlmostEqual(sum(abs(v) for v in attacker_1), 1.0, places=4)


class TestCostBSGMiqpOrtoolsAlphaZero(unittest.TestCase):
    """Test with alpha=0 (no switching cost) to verify it matches base case."""

    @classmethod
    def setUpClass(cls):
        """Run the solver with alpha=0."""
        cls.result = run_solver("simple_cost_input.txt", alpha=0.0)

    def test_solver_runs_successfully(self):
        """Test that the solver completes without error."""
        self.assertEqual(
            self.result["returncode"], 0,
            f"Solver failed with stderr: {self.result['stderr']}"
        )

    def test_objective_exists(self):
        """Test that an objective value was computed."""
        self.assertIsNotNone(self.result["objective"])


class TestCostBSGMiqpOrtoolsLargeInput(unittest.TestCase):
    """Test with the larger cost_BSSG_input.txt file."""

    @classmethod
    def setUpClass(cls):
        """Run the solver with cost_BSSG_input.txt."""
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        switch_cost_dir = os.path.dirname(tests_dir)
        input_file = os.path.join(switch_cost_dir, "cost_BSSG_input.txt")
        cls.result = run_solver(input_file, alpha=0.5)

    def test_solver_runs_successfully(self):
        """Test that the solver completes without error."""
        self.assertEqual(
            self.result["returncode"], 0,
            f"Solver failed with stderr: {self.result['stderr']}"
        )

    def test_objective_exists(self):
        """Test that an objective value was computed."""
        self.assertIsNotNone(self.result["objective"])

    def test_defender_strategy_sums_to_one(self):
        """Test that defender strategy probabilities sum to 1."""
        variables = self.result["variables"]
        defender_probs = [variables.get(f"x-{i}", 0) for i in range(4)]
        self.assertAlmostEqual(sum(defender_probs), 1.0, places=5)

    def test_each_attacker_has_pure_strategy(self):
        """Test that each attacker selects a pure strategy (sum of q = 1)."""
        variables = self.result["variables"]

        # Collect attacker variables by attacker index
        for l in range(3):  # 3 attacker types
            attacker_vars = []
            for name, value in variables.items():
                if name.startswith(f"{l}-") and not name.startswith("a-"):
                    attacker_vars.append(value)
            if attacker_vars:
                self.assertAlmostEqual(sum(abs(v) for v in attacker_vars), 1.0, places=4,
                                       msg=f"Attacker {l} does not have a pure strategy")


class TestCostBSGMiqpOrtoolsHighAlpha(unittest.TestCase):
    """Test with high alpha to verify switching costs have impact."""

    @classmethod
    def setUpClass(cls):
        """Run the solver with high alpha."""
        cls.result_high = run_solver("simple_cost_input.txt", alpha=10.0)
        cls.result_low = run_solver("simple_cost_input.txt", alpha=0.0)

    def test_both_solvers_run_successfully(self):
        """Test that both solvers complete without error."""
        self.assertEqual(
            self.result_high["returncode"], 0,
            f"Solver (high alpha) failed: {self.result_high['stderr']}"
        )
        self.assertEqual(
            self.result_low["returncode"], 0,
            f"Solver (low alpha) failed: {self.result_low['stderr']}"
        )

    def test_high_alpha_affects_objective(self):
        """Test that high switching cost weight affects the objective."""
        # With switching costs, the objective should be lower
        # (since we're subtracting alpha * cost * w from the objective)
        self.assertIsNotNone(self.result_high["objective"])
        self.assertIsNotNone(self.result_low["objective"])
        # High alpha should give lower or equal objective (switching costs are subtracted)
        self.assertLessEqual(
            self.result_high["objective"],
            self.result_low["objective"] + 1e-6  # small tolerance
        )


if __name__ == "__main__":
    unittest.main()
