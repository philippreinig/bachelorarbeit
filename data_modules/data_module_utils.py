import torch
import torchvision as tv
import torchvision.transforms.v2 as transform_lib

from akiset.akidataset import AKIDataset
from typing import Optional

class FilterVoidLabels(transform_lib.Transform):
    def __init__(self, valid_idx: int, void_idx: list[int], ignore_index: int) -> None:
        """Remove void classes from the label

        Args:
            classes (List[int]): List of all classes.
            void (List[int]): List of void classes.
            ignore_index (int): Replaces label of void classes
        """
        super().__init__()
        self.valid = valid_idx
        self.void = torch.as_tensor(void_idx)
        self.ignore = ignore_index

    def filter(self, label: torch.Tensor) -> torch.Tensor:
        """Replace void classes with ignore_index and
        renumber valid classes to [0, num_classes-1]"""
        label[torch.isin(label, self.void)] = self.ignore

        for new, old in enumerate(self.valid):
            label[label == old] = new

        return label

    def forward(self, image: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor]:
        """Replace void classes with ignore class and renumber valid classes to
        [0, num_classes-1]. Implemented in a backwards compatible way suppporting both
        legacy and v2 transform interfaces.

        Args:
            flat_inputs: Tuple containing image and label with length=2.
                image (torch.Tensor): image tensor with shape [C, H, W].
                label (torch.Tensor): corresponding label with values in [0, 33].
        Returns:
            image, label: Pass through image and return filtered label
        """
        with tv.tv_tensors.set_return_type("TVTensor"):
            return image, self.filter(label)

def elems_in_dataloader(dataset_size: int, limit: Optional[int] = None) -> int:
    return min(dataset_size, limit) if limit else dataset_size
    
def runs_per_epoch(dataset_size: int, batch_size: int, limit: Optional[int] = None) -> int:
    dataloader_size = elems_in_dataloader(dataset_size, limit)
    if dataloader_size % batch_size == 0:
        return int(dataloader_size / batch_size)
    else:
        return int(dataloader_size / batch_size) + 1
    

def get_label_distribution(ds: AKIDataset) -> tuple[int, int]:
    sunny_imgs = 0
    rainy_imgs = 0

    for elem in ds:
        if elem[1] == "sunny":
            sunny_imgs += 1
        elif elem[1] == "rain":
            rainy_imgs += 1
        else:
            raise ValueError(f"Unknown label: {elem[1]}")

    return sunny_imgs, rainy_imgs