import torch
from fusion.clm import ns

def fuse_unimodal_with_clm(predictions: torch.Tensor, clm: torch.Tensor) -> torch.Tensor:
    clm_ns = clm * ns(clm)
    assert(torch.allclose(clm_ns.sum(dim=1), torch.ones(clm.shape[0]))), f"Expected sum of each row in clm_ns to be 1, but they are: {clm_ns.sum(dim=1)}"
    fused_logits = predictions @ clm_ns
    norms = fused_logits.sum(dim=1, keepdim=True)
    fused_predictions = fused_logits / norms

    assert(torch.allclose(fused_predictions.sum(dim=1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of each row in fused_predictions to be 1, but they are: {fused_predictions.sum(dim=1)}"

    return fused_predictions


def fuse_predictions(cam_predictions: torch.Tensor, lid_predictions: torch.Tensor, clm: torch.Tensor) -> torch.Tensor:
    assert(cam_predictions.shape == lid_predictions.shape), f"Shapes of camera and lidar predictions must be equal, but they are {cam_predictions}, {lid_predictions}"
    amt_classes = [cam_predictions.shape[-1]]

    fused_predictions = torch.einsum('ic, il, rcl -> ir', cam_predictions, lid_predictions, clm)

    assert(fused_predictions.shape == [1, amt_classes]), f"Expected fused predictions to be of shape [1, {amt_classes}], but it is {fuse_predictions.shape}"
    assert(fused_predictions.sum() == 1), f"Expected sum of fused predictions to be 1, but is {fuse_predictions.sum()}"

    return fused_predictions