import torch
import logging

from fusion.clm import ns
from typing import Optional

log = logging.getLogger(__name__)


def fuse_unimodal_with_clm(predictions: torch.Tensor, scenario: str, clm_rain: Optional[torch.Tensor], clm_sun: Optional[torch.Tensor], rain_labels: Optional[torch.BoolTensor]) -> torch.Tensor:
    if scenario == "rain":
        clm_rain_ns = clm_rain * ns(clm_rain)
        assert(torch.allclose(clm_rain_ns.sum(dim=1), torch.ones(clm_rain.shape[0]))), f"Expected sum of each row in clm_ns to be 1, but they are: {clm_rain_ns.sum(dim=1)}"

        fused_logits = predictions @ clm_rain_ns
        norms = fused_logits.sum(dim=1, keepdim=True)
        fused_predictions = fused_logits / norms

        assert(torch.allclose(fused_predictions.sum(dim=1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of each row in fused_predictions to be 1, but they are: {fused_predictions.sum(dim=1)}"

        return fused_predictions

    elif scenario == "sun":
        clm_sun_ns = clm_sun * ns(clm_sun)
        assert(torch.allclose(clm_sun_ns.sum(dim=1), torch.ones(clm_sun.shape[0]))), f"Expected sum of each row in clm_ns to be 1, but they are: {clm_sun_ns.sum(dim=1)}"
    
        fused_logits = predictions @ clm_sun_ns
        norms = fused_logits.sum(dim=1, keepdim=True)
        fused_predictions = fused_logits / norms

        assert(torch.allclose(fused_predictions.sum(dim=1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of each row in fused_predictions to be 1, but they are: {fused_predictions.sum(dim=1)}"

        return fused_predictions

    elif scenario == "combined":
        clm_rain_ns = clm_rain * ns(clm_rain)
        clm_sun_ns = clm_sun * ns(clm_sun)

        assert(torch.allclose(clm_rain_ns.sum(dim=1), torch.ones(clm_rain.shape[0]))), f"Expected sum of each row in clm_ns to be 1, but they are: {clm_rain_ns.sum(dim=1)}"
        assert(torch.allclose(clm_sun_ns.sum(dim=1), torch.ones(clm_sun.shape[0]))), f"Expected sum of each row in clm_ns to be 1, but they are: {clm_sun_ns.sum(dim=1)}"   

        fused_logits_list = []

        counter_fused_for_rain_scenario = 0
        counter_fused_for_sun_scenario = 0

        for i, prediction in enumerate(predictions):
            if rain_labels[i // (800*1600)].item() == 1:
                fused_logits_list.append(prediction @ clm_rain_ns)
                counter_fused_for_rain_scenario += 1
            else:
                fused_logits_list.append(prediction @ clm_sun_ns)
                counter_fused_for_rain_scenario += 1

        log.info(f"Fused {counter_fused_for_rain_scenario} predictions for rain scenario and {counter_fused_for_sun_scenario} predictions for sun scenario")
        
        fused_logits = torch.stack(fused_logits_list)
        norms = fused_logits.sum(dim=1, keepdim=True)
        fused_predictions = fused_logits / norms

        assert(torch.allclose(fused_predictions.sum(dim=1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of each row in fused_predictions to be 1, but they are: {fused_predictions.sum(dim=1)}"
        
        return fused_predictions


def fuse_unimodal_with_rain_probs(sis_preds_rshpd: torch.tensor, rain_probabilities: torch.tensor, clm_rain: torch.tensor, clm_sun: torch.tensor):
    clm_ns_rain = clm_rain * ns(clm_rain)
    clm_ns_sun = clm_sun * ns(clm_sun)

    fused_logits = []

    for i in range(len(sis_preds_rshpd)):
        sis_preds = rain_probabilities[i] * (sis_preds_rshpd[i] @ clm_ns_rain) + (1 - rain_probabilities[i]) * (sis_preds_rshpd[i] @ clm_ns_sun)
        fused_logits.append(sis_preds)

    fused_logits = torch.cat(fused_logits, dim=0)
    norms = fused_logits.sum(dim=1, keepdim=True)

    fused_predictions = fused_logits / norms

    assert(torch.allclose(fused_predictions.sum(dim=1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of each row in fused_predictions to be 1, but they are: {fused_predictions.sum(dim=1)}"

    return fused_predictions

def fuse_multimodal(cam_preds: torch.Tensor, lid_preds: torch.Tensor, clm: torch.Tensor):
    assert cam_preds.shape == lid_preds.shape, f"Shapes of camera and lidar predictions must be equal, but they are {cam_preds.shape}, {lid_preds.shape}"
    assert cam_preds.shape[-1] == lid_preds.shape[-1] == clm.shape[0],  f"Amount of predicted classes must be the same for camera and lidar predictions as well as the CLM, but are {cam_preds.shape[-1]}, {lid_preds.shape[-1]}, {clm.shape[0]}"
    assert clm.shape[0] == clm.shape[1] == clm.shape[2], f"CLM rain must be cubic tensor with shape [c, c, c], but is: {clm.shape}"

    fused_predictions = torch.einsum('ic, il, rcl -> ir', cam_preds, lid_preds, clm)
    fused_predictions = fused_predictions / fused_predictions.sum(dim=-1, keepdim=True)
    
    assert(fused_predictions.shape == cam_preds.shape == lid_preds.shape), f"Expected fused predictions to be of shape {cam_preds.shape}, but it is {fused_predictions.shape}"
    assert(torch.allclose(fused_predictions.sum(dim=-1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of last dimension across fused predictions to be 1, but is {fused_predictions.sum(dim=-1)}"

    return fused_predictions

def fuse_multimodal_with_rain_probs(cam_preds: torch.Tensor, lid_preds: torch.Tensor, rain_probs: torch.Tensor, clm_rain: torch.Tensor, clm_sun: torch.Tensor):
    assert cam_preds.shape == lid_preds.shape, f"Shapes of camera and lidar predictions must be equal, but they are {cam_preds.shape}, {lid_preds.shape}"
    assert cam_preds.shape[0] == lid_preds.shape[0] == rain_probs.shape[0], f"Amount of predictions must be the same for camera, lidar and rain predictions, but are {cam_preds.shape[0]}, {lid_preds.shape[0]}, {rain_probs.shape[0]}"
    assert clm_rain.shape[0] == clm_rain.shape[1] == clm_rain.shape[2], f"CLM rain must be cubic tensor with shape [c, c, c], but is: {clm_rain.shape}"
    assert clm_sun.shape[0] == clm_sun.shape[1] == clm_sun.shape[2], f"CLM sun must be cubic tensor with shape [c, c, c], but is: {clm_sun.shape}"
    assert cam_preds.shape[-1] == lid_preds.shape[-1] == clm_rain.shape[0] == clm_sun.shape[0],  f"Amount of predicted classes must be the same for camera and lidar predictions as well as the CLMs, but are {cam_preds.shape[-1], lid_preds.shape[-1], clm_rain[0].shape, clm_sun[0].shape}"

    fused_predictions = rain_probs.unsqueeze(1) * torch.einsum('ic, il, rcl -> ir', cam_preds, lid_preds, clm_rain) + ((torch.ones_like(rain_probs) - rain_probs).unsqueeze(1)) * torch.einsum('ic, il, rcl -> ir', cam_preds, lid_preds, clm_sun)
    fused_predictions = fused_predictions / fused_predictions.sum(dim=-1, keepdim=True)

    assert(fused_predictions.shape == cam_preds.shape == lid_preds.shape), f"Expected fused predictions to be of shape {cam_preds.shape}, but it is {fused_predictions.shape}"
    assert(torch.allclose(fused_predictions.sum(dim=-1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of last dimension across fused predictions to be 1, but is {fused_predictions.sum(dim=-1)}"

    return fused_predictions

def fuse_predictions(cam_predictions: torch.Tensor, lid_predictions: torch.Tensor, clm: torch.Tensor) -> torch.Tensor:
    assert(cam_predictions.shape == lid_predictions.shape), f"Shapes of camera and lidar predictions must be equal, but they are {cam_predictions}, {lid_predictions}"

    fused_predictions = torch.einsum('ic, il, rcl -> ir', cam_predictions, lid_predictions, clm)

    assert(fused_predictions.shape == cam_predictions.shape == lid_predictions.shape), f"Expected fused predictions to be of shape {cam_predictions.shape}, but it is {fused_predictions.shape}"
    assert(torch.allclose(fused_predictions.sum(dim=-1), torch.ones(fused_predictions.shape[0]))), f"Expected sum of last dimension across fused predictions to be 1, but is {fused_predictions.sum(dim=-1)}"

    return fused_predictions