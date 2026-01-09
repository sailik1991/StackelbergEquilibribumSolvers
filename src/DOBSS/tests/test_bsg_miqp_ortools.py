#!/usr/bin/python
"""
Test for BSG_miqp_ortools.py solver.
Runs the solver with input.txt and verifies expected output.
"""

import subprocess
import os
import unittest

class TestBSGMiqpOrtools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run the solver once and parse output."""
        # Get the DOBSS directory (parent of tests/)
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        dobss_dir = os.path.dirname(tests_dir)
        solver_path = os.path.join(dobss_dir, "BSG_miqp_ortools.py")
        input_path = os.path.join(dobss_dir, "input.txt")

        result = subprocess.run(
            ["python3", solver_path, input_path],
            capture_output=True,
            text=True,
            cwd=dobss_dir
        )

        cls.stdout = result.stdout
        cls.stderr = result.stderr
        cls.returncode = result.returncode

        # Parse output
        cls.variables = {}
        cls.objective = None

        for line in cls.stdout.split("\n"):
            if line.startswith("Obj -> "):
                cls.objective = float(line.split(" -> ")[1].strip())
            elif " -> " in line:
                parts = line.split(" -> ")
                name = parts[0].strip()
                value = float(parts[1].strip())
                cls.variables[name] = value

    def test_solver_runs_successfully(self):
        """Test that the solver completes without error."""
        self.assertEqual(self.returncode, 0, f"Solver failed with stderr: {self.stderr}")

    def test_objective_value(self):
        """Test that the objective value is correct."""
        self.assertIsNotNone(self.objective)
        self.assertAlmostEqual(self.objective, 0.912143, places=4)

    def test_defender_strategy_sums_to_one(self):
        """Test that defender strategy probabilities sum to 1."""
        defender_probs = [
            self.variables.get("x-0", 0),
            self.variables.get("x-1", 0),
            self.variables.get("x-2", 0),
            self.variables.get("x-3", 0),
        ]
        self.assertAlmostEqual(sum(defender_probs), 1.0, places=5)

    def test_defender_strategy_values(self):
        """Test expected defender strategy values."""
        self.assertAlmostEqual(self.variables["x-0"], 0.428571, places=4)
        self.assertAlmostEqual(self.variables["x-1"], 0.414286, places=4)
        self.assertAlmostEqual(self.variables["x-2"], 0.0, places=4)
        self.assertAlmostEqual(self.variables["x-3"], 0.157143, places=4)

    def test_attacker_0_response(self):
        """Test that attacker 0 chooses Attack2."""
        self.assertAlmostEqual(self.variables["0-Attack1"], 0.0, places=4)
        self.assertAlmostEqual(self.variables["0-Attack2"], 1.0, places=4)
        self.assertAlmostEqual(self.variables["0-Attack9"], 0.0, places=4)

    def test_attacker_1_response(self):
        """Test that attacker 1 chooses Attack1+Attack4."""
        self.assertAlmostEqual(self.variables["1-Attack1"], 0.0, places=4)
        self.assertAlmostEqual(self.variables["1-Attack4"], 0.0, places=4)
        self.assertAlmostEqual(self.variables["1-Attack1+Attack4"], 1.0, places=4)
        self.assertAlmostEqual(self.variables["1-Attack9"], 0.0, places=4)

    def test_attacker_2_response(self):
        """Test that attacker 2 chooses Attack2+Attack3."""
        self.assertAlmostEqual(self.variables["2-Attack2+Attack3"], 1.0, places=4)

    def test_attacker_3_response(self):
        """Test that attacker 3 chooses Attack3."""
        self.assertAlmostEqual(self.variables["3-Attack3"], 1.0, places=4)

    def test_each_attacker_has_pure_strategy(self):
        """Test that each attacker chooses exactly one pure strategy."""
        # Attacker 0
        attacker_0 = [self.variables.get(f"0-Attack1", 0),
                      self.variables.get(f"0-Attack2", 0),
                      self.variables.get(f"0-Attack9", 0)]
        self.assertAlmostEqual(sum(abs(v) for v in attacker_0), 1.0, places=4)

        # Attacker 1
        attacker_1 = [self.variables.get(f"1-Attack1", 0),
                      self.variables.get(f"1-Attack4", 0),
                      self.variables.get(f"1-Attack1+Attack4", 0),
                      self.variables.get(f"1-Attack9", 0)]
        self.assertAlmostEqual(sum(abs(v) for v in attacker_1), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
