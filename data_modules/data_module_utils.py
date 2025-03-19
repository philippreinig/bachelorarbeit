import torch
import torchvision.transforms.v2 as transform_lib

from torchvision import tv_tensors
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

    def forward(self, image: torch.Tensor, label: torch.Tensor, weather_condition: torch.Tensor) -> tuple[torch.Tensor]:
        """Replace void classes with ignore class and renumber valid classes to
        [0, num_classes-1]. Implemented in a backwards compatible way suppporting both
        legacy and v2 transform interfaces.

        Args:
            flat_inputs: Tuple containing image and label with length=2.
                image (torch.Tensor): image tensor with shape [C, H, W].
                label (torch.Tensor): corresponding label with values in [0, 33].
        Returns:
            image, label: Pass through image and weather_condition and return filtered label
        """
        image = tv_tensors.Image(image)
        label = tv_tensors.Mask(self.filter(label))

        return image, label, weather_condition

def elems_in_dataset(dataset_size: int, limit: Optional[int] = None) -> int:
    return min(dataset_size, limit) if limit else dataset_size
    
def runs_per_epoch(dataset_size: int, batch_size: int, limit: Optional[int] = None) -> int:
    dataloader_size = elems_in_dataset(dataset_size, limit)
    if dataloader_size % batch_size == 0:
        return int(dataloader_size / batch_size)
    else:
        return int(dataloader_size / batch_size) + 1
    

def get_label_distribution(ds: AKIDataset, weather_indx: int) -> tuple[int, int]:
    sunny_imgs = 0
    rainy_imgs = 0

    for elem in ds:
        if elem[weather_indx] == "sunny":
            sunny_imgs += 1
        elif elem[weather_indx] == "rain":
            rainy_imgs += 1
        else:
            raise ValueError(f"Unknown label: {elem[1]}")

    return sunny_imgs, rainy_imgs

def calc_waymo_rainy_vs_sunny_image_stats(ds: AKIDataset) -> tuple[float, float, float, float]:
    # Initialize counters and accumulators
    sunny_mean = torch.zeros(3)
    sunny_M2 = torch.zeros(3)
    sunny_pixels = 0

    rainy_mean = torch.zeros(3)
    rainy_M2 = torch.zeros(3)
    rainy_pixels = 0

    for elem in ds:
        img, lbl = elem
        img = img.view(3, -1)  # Flatten spatial dimensions
        batch_mean = img.mean(dim=1)
        batch_var = img.var(dim=1, unbiased=False)
        batch_pixels = img.shape[1]

        if lbl == 0:  # Sunny images
            delta = batch_mean - sunny_mean
            sunny_mean += delta * (batch_pixels / (sunny_pixels + batch_pixels))
            sunny_M2 += batch_var * batch_pixels + delta**2 * (sunny_pixels * batch_pixels) / (sunny_pixels + batch_pixels)
            sunny_pixels += batch_pixels

        else:  # Rainy images
            delta = batch_mean - rainy_mean
            rainy_mean += delta * (batch_pixels / (rainy_pixels + batch_pixels))
            rainy_M2 += batch_var * batch_pixels + delta**2 * (rainy_pixels * batch_pixels) / (rainy_pixels + batch_pixels)
            rainy_pixels += batch_pixels

    sunny_std = torch.sqrt(sunny_M2 / sunny_pixels)
    rainy_std = torch.sqrt(rainy_M2 / rainy_pixels)

    return sunny_mean, sunny_std, rainy_mean, rainy_std


def randomly_crop(image: torch.Tensor, segmentation_mask: torch.Tensor, crop_size: tuple[int, int]):
    """Randomly crop image and segmentation mask to crop_size and return the indices applied.

    Args:
        image (torch.Tensor): Image tensor with shape [C, H, W].
        segmentation_mask (torch.Tensor): Segmentation mask tensor with shape [H, W].
        crop_size (tuple[int, int]): Tuple containing crop width and height.

    Returns:
        torch.Tensor, torch.Tensor: Cropped image, segmentation mask and indices.
    """
    i, j, h, w = transform_lib.RandomCrop.get_params(image, output_size=crop_size)
    image = transform_lib.functional.crop(image, i, j, h, w)
    segmentation_mask = transform_lib.functional.crop(segmentation_mask, i, j, h, w)

    return image, segmentation_mask, i, j, h, w

def weather_condition2numeric(weather_condition: str) -> list:
    mapping = {
        "Clear Sky": 0,
        "Heavy Rain": 1,
        "Dense Drizzle": 1,
        "Light Drizzle": 1,
        "Light Rain": 1,
        "Mainly Clear": 0,
        "Moderate Drizzle": 1,
        "Moderate Rain": 1,
        "Overcast": 0,
        "Partly Cloudy": 0,
        "rain": 1,
        "sunny": 0
    }

    if "Snow" in weather_condition:
        raise ValueError("Can't embed snow!")

    return mapping[weather_condition]

