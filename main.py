import torch
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from torch.utils.data import DataLoader
import uuid

from akiset import AKIDataset

IMGS_FOLDER = "imgs/"


def is_full_hd_img(img: torch.Tensor) -> bool:
    return img[0].size()[-1] == 1920 and img[0].size()[1] == 1280


def get_colored_segmentation(segmentation: torch.Tensor) -> np.ndarray:
    # Define a simple color map: as example, below is a random color map for 20 classes.
    colors = np.random.randint(0, 255, (30, 3), dtype=np.uint8)
    colored_segmentation = colors[segmentation.cpu().numpy()]
    return colored_segmentation

def count_elements(dataset):
    return sum(1 for _ in dataset)

def overlay_segmentation(img: np.ndarray, segmentation: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlayed_img = img * (1 - alpha) + segmentation * alpha
    return overlayed_img.astype(np.uint8)


def divide_batch_of_tensors(t: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """
    Divide tensor into a grid of n rows and m columns.

    raise ValueError if width of tensor is not divisible by n or height of tensor is not divisible by m.
    """
    if len(t.shape) != 4:
        raise ValueError(f"Input tensor must have 4 dimensions (b x l x w x h), but shape is:  {t.shape}")

    b, l, w, h = t.shape

    if l != 3:
        raise ValueError(f"Expected 3 layers, but got: {l}")

    if w % m != 0:
        raise ValueError(f"Width of image {w} is not divisible by {m}")

    if h % n != 0:
        raise ValueError(f"Height of image {h} is not divisible by {n}")

    cell_width = w // m
    cell_height = h // n

    new_tensor = torch.zeros((b, n * m, l, cell_height, cell_width))

    for i in range(n):
        for j in range(m):
            new_tensor[:, i * m + j, :, :, :] = t[:, :, :, i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

    return new_tensor

def try_to_do_some_stuff_with_dataset():
    train_ds = AKIDataset(data={"camera": ["image"], "camera_segmentation": ["camera_segmentation"]},
                          splits=["training"],
                          datasets=['nuimages'],
                          scenario='rain',
                          offset=0)
    print(f"Number of elements in train_ds: {count_elements(train_ds)}")

    filtered_train_ds = train_ds.filter(filter_fn=is_full_hd_img)

    # Print the number of elements in train_ds
    print(f"Number of elements in filtered_train_ds: {count_elements(train_ds)}")

    data_loader = DataLoader(filtered_train_ds, batch_size=9)

    print("==== Dataset loaded =====")

    for batch in data_loader:
        print(f"Batch is of type: {type(batch)}")
        imgs, segmentations = batch[0], batch[1]

        fig, axs = plt.subplots(3,3)
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            img = imgs[i - 1]
            segmentation = segmentations[i - 1]

            img = img.permute(1, 2, 0).numpy()

            colored_segmentation = get_colored_segmentation(segmentation)
            overlayed_img = overlay_segmentation(img, colored_segmentation)

            ax = axs[i // cols - 1, i % cols - 1]
            ax.axis("off")
            ax.imshow(img)
            ax.imshow(overlayed_img, alpha=0.7)
        plt.tight_layout(pad=0)
        plt.savefig(f"{IMGS_FOLDER + str(uuid.uuid4())}")
        plt.close(fig)


if __name__ == "__main__":
    try_to_do_some_stuff_with_dataset()
