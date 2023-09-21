import unittest

from rldmpc.cvxpy_system.evaluation import LtiNetwork


class TestStageCost(unittest.TestCase):
    def test_without_violation(self):
        LtiNetwork()

    def test_with_violation(self):
        pass


if __name__ == "__main__":
    unittest.main()
