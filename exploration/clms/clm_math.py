import torch

import logging

log = logging.getLogger("rich")

def calc_clm(predictions: torch.Tensor, labels: torch.Tensor):
       log.info(f"Predictions shape: {predictions.shape}")
       log.info(f"Labels shape: {labels.shape}")
       probabilistic_transition_matrices = torch.bmm(predictions, labels)
       clm = torch.sum(probabilistic_transition_matrices, 0)
       return clm

def calc_normalized_clm(predictions: torch.tensor, labels: torch.Tensor):

       # TODO: Implement validity checks
       log.info(f"Predictions shape: {predictions.shape}")
       log.info(f"Labels shape: {labels.shape}")

       batch_size = predictions.shape[0]

       clm = torch.sum(torch.bmm(predictions, labels), 0)
       clm_normalized = clm / batch_size

       return clm_normalized