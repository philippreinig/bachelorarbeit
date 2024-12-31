import unittest
import torch

import logging
from rich.logging import RichHandler

logging.basicConfig(level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])

log = logging.getLogger("rich")

from clm_math import calc_clm, calc_normalized_clm


class TestCLMCalculations(unittest.TestCase):

    def test_calc_clm(self):
        predictions = torch.Tensor([[[0.2], [0.5], [0.3]],
                              [[0.4], [0.3], [0.3]],
                              [[0.1], [0.6], [0.3]],
                              [[0.4], [0.4], [0.2]],
                              [[0.2], [0.2], [0.6]],
                              [[0.5], [0.3], [0.2]],
                              [[0.1], [0.7], [0.2]],
                              [[0.3], [0.2], [0.5]],
                              [[0.4], [0.5], [0.1]],
                              [[0.2], [0.3], [0.5]]])

        labels = torch.Tensor([[[0, 1, 0]],
                               [[1, 0, 0]],
                               [[0, 1, 0]],
                               [[0, 1, 0]],
                               [[0, 0, 1]],
                               [[1, 0, 0]],
                               [[0, 1, 0]],
                               [[0, 0, 1]],
                               [[1, 0, 0]],
                               [[0, 0, 1]]])

        expected_clm = torch.Tensor([[1.3, 0.8, 0.7], [1.1, 2.2, 0.7], [0.6, 1.0, 1.6]])
        expected_clm_normalized = torch.Tensor([[0.13, 0.08, 0.07], [0.11, 0.22, 0.07], [0.06, 0.1, 0.16]])

        clm = calc_clm(predictions, labels)
        clm_normalized = calc_normalized_clm(predictions, labels)

        log.info(f"Expected clm: {expected_clm}")
        log.info(f"Actual clm: {clm}")

        log.info(f"Expected normalized clm: {expected_clm_normalized}")
        log.info(f"Actual normalized clm: {clm_normalized}")

        log.info(torch.isclose(expected_clm, clm))
        log.info(torch.isclose(expected_clm_normalized, clm_normalized))

        self.assertTrue(torch.allclose(expected_clm, clm))
        self.assertTrue(torch.allclose(expected_clm_normalized, clm_normalized))


if __name__ == '__main__':
    unittest.main()