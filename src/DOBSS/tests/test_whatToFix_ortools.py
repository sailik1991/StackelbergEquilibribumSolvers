#!/usr/bin/python
"""
Test for whatToFix_ortools.py solver.
Runs the solver with input.txt and verifies expected output.
"""

import subprocess
import os
import unittest


class TestWhatToFixOrtools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Run the solver once and parse output."""
        # Get the DOBSS directory (parent of tests/)
        tests_dir = os.path.dirname(os.path.abspath(__file__))
        dobss_dir = os.path.dirname(tests_dir)
        solver_path = os.path.join(dobss_dir, "whatToFix_ortools.py")
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
        cls.best_obj = None
        cls.best_attacks = []
        cls.all_results = []

        lines = cls.stdout.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Best Obj value -> "):
                cls.best_obj = float(line.split(" -> ")[1].strip())
            elif line.startswith("[("):
                # Parse the allSet list
                import ast
                cls.all_results = ast.literal_eval(line.strip())
            elif line.startswith("('") and line.endswith("')") or line.startswith("('") and line.endswith("',)"):
                # Parse best attack tuples
                import ast
                cls.best_attacks.append(ast.literal_eval(line.strip()))

    def test_solver_runs_successfully(self):
        """Test that the solver completes without error."""
        self.assertEqual(self.returncode, 0, f"Solver failed with stderr: {self.stderr}")

    def test_best_objective_value(self):
        """Test that the best objective value is correct."""
        self.assertIsNotNone(self.best_obj)
        self.assertAlmostEqual(self.best_obj, 2.514286, places=4)

    def test_best_attack_to_remove(self):
        """Test that removing Attack3 gives the best result."""
        self.assertIn(('Attack3',), self.best_attacks)

    def test_all_attacks_evaluated(self):
        """Test that all 9 attack combinations were evaluated."""
        self.assertEqual(len(self.all_results), 9)

    def test_attack9_removal_value(self):
        """Test objective value when Attack9 is removed."""
        attack9_result = next((obj for attacks, obj in self.all_results if attacks == ('Attack9',)), None)
        self.assertIsNotNone(attack9_result)
        self.assertAlmostEqual(attack9_result, 1.3175, places=4)

    def test_attack1_removal_value(self):
        """Test objective value when Attack1 is removed."""
        attack1_result = next((obj for attacks, obj in self.all_results if attacks == ('Attack1',)), None)
        self.assertIsNotNone(attack1_result)
        self.assertAlmostEqual(attack1_result, 0.344643, places=4)

    def test_attack3_gives_highest_value(self):
        """Test that Attack3 removal gives the highest objective among all."""
        attack3_result = next((obj for attacks, obj in self.all_results if attacks == ('Attack3',)), None)
        self.assertIsNotNone(attack3_result)
        # Verify it's the maximum
        max_obj = max(obj for _, obj in self.all_results)
        self.assertAlmostEqual(attack3_result, max_obj, places=4)


if __name__ == "__main__":
    unittest.main()
