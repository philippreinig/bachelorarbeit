import torch

def create_unimodal_normalized_clm_from_img_batch(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """_summary_
    Args:
        predictions (torch.Tensor): Tensor of shape [n, h, w, p] containing sis predictions for n images of
        size [h,w] over p classes.
        labels (torch.Tensor): Tensor of shape [n, h, w, l] containing one-hot encoded labels for
        n images of size [h,w] over p classes => l e [0,p[
    """
    predictions_n, predictions_h, predictions_w, predictions_p = predictions.shape
    labels_n, labels_h, labels_w, labels_l = labels.shape
    assert(predictions.dim() == 4), f"Predictions must have 4 dimensions because required shape is: [n,h,w,p]. Passed shape is: {predictions.shape}"
    assert(labels.dim() == 4), f"Labels must have 4 dimensions because required shape is: [n,h,w,l]. Passed shape is: {labels.shape}"
    assert(predictions_n == labels_n), f"Batch sizes between predictions and labels don't match: {predictions_n}, {labels_n}"
    assert(predictions_h == labels_h), f"Height of image doesn't match between predictions and labels: {predictions_h}, {labels_h}"
    assert(predictions_w == labels_w), f"Width of image doesn't match between predictions and labels: {predictions_w}, {labels_w}"
    assert (labels.sum(dim=-1) == 1).all() and ((labels == 0) | (labels == 1)).all(), "Labels must be one-hot encoded (only 0s and 1s, summing to 1 along the last dimension)"
    assert torch.allclose(predictions.sum(dim=-1), torch.ones_like(predictions[..., 0]), atol=1e-5), "Predictions must sum to approximately 1 across the last dimension"
    
    amt_elements = predictions_n * predictions_h * predictions_w
    clm = torch.einsum("nhwp,nhwl->pl", predictions, labels)
    clm_normalized = torch.div(clm, amt_elements)

    return clm_normalized

def create_unimodal_normalized_clm(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """_summary_
    Args:
        predictions (torch.Tensor): Tensor of shape [..., c] containing predictions over c classes.
        labels (torch.Tensor): Tensor of shape [..., l] containing one-hot encoded labels over c classes => l e [0,c[
        Shape of labels must match shape of predictions
    """
    assert(predictions.dim() >= 2), "Predictions must have at least 2 dimensions because required shape is: [...,c]"
    assert(labels.shape == predictions.shape), "Labels must have same shapes as predictions"
    assert (labels.sum(dim=-1) == 1).all() and ((labels == 0) | (labels == 1)).all(), "Labels must be one-hot encoded (only 0s and 1s, summing to 1 along the last dimension)"
    assert torch.allclose(predictions.sum(dim=-1), torch.ones_like(predictions[..., 0]), atol=1e-5), "Predictions must sum to 1 across the last dimension"

    clm = torch.einsum("...p,...l->pl", predictions, labels)
    amt_elements = torch.prod(torch.tensor([predictions.shape[0:-1]]))
    clm_normalized = torch.div(clm, amt_elements)

    return clm_normalized

def ns(clm: torch.Tensor) -> torch.Tensor:
    assert(clm.dim() == 2), f"Passed matrix must be two dimensional, but has shape: {clm.shape}"
    assert(clm.shape[0] == clm.shape[1]), f"Passed matrix must be quadratic, but has shape: {clm.shape}"
    row_sums = clm.sum(dim=1, keepdim=True)
    ns = 1 / row_sums
    # If all values in a row are 0, a 0 by 0 divison occurs by the expression above -> change all nan values to 0
    ns = torch.nan_to_num(ns, nan=0) 
    return ns

def nr(clm: torch.Tensor) -> torch.Tensor:
    assert(clm.dim() == 2), f"Passed matrix must be two dimensional, but has shape: {clm.shape}"
    assert(clm.shape[0] == clm.shape[1]), f"Passed matrix must be quadratic, but has shaep: {clm.shape}"
    col_sums = clm.sum(dim=0)  
    nr = 1 / col_sums
    # If all values in a column are 0, a 0 by 0 divison occurs by the expression above -> change all nan values to 0
    nr = torch.nan_to_num(nr, nan=0)   
    return nr

def pr(clm: torch.Tensor) -> torch.Tensor:
    return clm.sum(dim=1)

def ps(clm: torch.Tensor) -> torch.Tensor:
    return clm.sum(dim=0)

def create_multimodal_normalized_clm(cam_predictions: torch.Tensor, lid_predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ Create the 3D multimodal CLM

    Args:
        cam_predictions (torch.Tensor): A [..., p_cam] tensor of camera predictions
        lid_predictions (torch.Tensor): A [..., p_lid] tensor of lidar predictions with same shape as cam_predictions
        labels (torch.Tensor): A [One-hot encoded labels 

    Returns:
        torch.Tensor: _description_
    """
    assert(cam_predictions.shape == lid_predictions.shape), f"Camera and lidar predictions must have same shape, but are: {cam_predictions.shape}, {lid_predictions.shape}"
    assert(cam_predictions.shape == labels.shape), f"Predictions and labels must have same shape, but are: {cam_predictions.shape}, {labels.shape}"

    amt_classes = cam_predictions.shape[-1]

    clm_cam = create_unimodal_normalized_clm(cam_predictions, labels)
    nr_cam = nr(clm_cam)
    clm_lid = create_unimodal_normalized_clm(lid_predictions, labels)
    nr_lid = nr(clm_lid)
    p_cam_if_real = clm_cam * nr_cam # P(S^C|R)
    p_lid_if_real = clm_lid * nr_lid # P(S^L|R)
    
    # Obtaining P(R) from clm_cam because P(R|^{C,L}) unknown before obtaining resulting matrix of this function,
    # but P(R)=P(R^C)=P(R^L) if input predictions are only from fusable regions/pixels  
    p_r = pr(clm_cam)

    # Create multimodal CLM
    p_r_sc_sl = torch.einsum('r,rc,rl->rcl', p_r, p_cam_if_real, p_lid_if_real) # Numerator of (Rost, eq. 4.9)
    p_r_sc_sl = p_r_sc_sl / p_r_sc_sl.sum(0)                                    # Denominator of (Rost, eq. 4.9)

    assert(p_r_sc_sl.shape == [amt_classes, amt_classes, amt_classes])

    return p_r_sc_sl

def check_create_clm():
    b, h, w, c = 2, 1, 3, 5  # Example dimensions
    cam_logits = torch.randn(b, h, w, c)
    cam_predictions = torch.nn.functional.softmax(cam_logits, dim=-1)
    lid_logits = torch.randn(b, h, w, c)
    lid_predictions = torch.nn.functional.softmax(lid_logits, dim=-1)

    labels = torch.randint(0, c, (b, h, w))  # Shape: (b, h, w)
    # Convert to one-hot encoding
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=c).float()

    print(f"Camera predictions tensor: {cam_predictions}")
    print(f"Label tensor: {labels_one_hot}")

    cam_clm_from_img_batch = create_unimodal_normalized_clm_from_img_batch(cam_predictions, labels_one_hot)
    cam_clm_universal = create_unimodal_normalized_clm(cam_predictions, labels_one_hot)
    lid_clm = create_unimodal_normalized_clm(lid_predictions, labels_one_hot)

    print(f"From img batch: {cam_clm_from_img_batch}")
    print(f"Universal: {cam_clm_universal}")

    print(f"Sum of normalized CLM: {cam_clm_from_img_batch.sum()}")

    print(torch.allclose(cam_clm_from_img_batch, cam_clm_universal))

    print(f"Multimodal CLM: {create_multimodal_normalized_clm(cam_predictions, lid_predictions, labels_one_hot)}")

if __name__ == "__main__":
    check_create_clm()