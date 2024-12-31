import torch
import numpy as np
from akiset.db_helpers import get_table_graph, setup_tables
from matplotlib import pyplot as plt
import networkx as nx
from psycopg import connect
from torchvision.utils import draw_segmentation_masks
from torchvision import tv_tensors
import torchvision.transforms.v2 as transform_lib
from data import get_aki_label_colors

import logging
log = logging.getLogger("rich")


def is_full_hd_img(img: torch.Tensor) -> bool:
    return img[0].size()[-1] == 1920 and img[0].size()[1] == 1280


def get_colored_segmentation(segmentation: torch.Tensor) -> np.ndarray:
    # Define a simple color map: as example, below is a random color map for 20 classes.
    colors = np.random.randint(0, 255, (30, 3), dtype=np.uint8)
    colored_segmentation = colors[segmentation.cpu().numpy()]
    return colored_segmentation

def overlay_segmentation(img: np.ndarray, segmentation: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlayed_img = img * (1 - alpha) + segmentation * alpha
    return overlayed_img.astype(np.uint8)


def divide_batch_of_tensors(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Divide tensor into a grid of n rows and m columns.

    raise ValueError if width of tensor is not divisible by n or height of tensor is not divisible by m.
    """
    if len(t.shape) != 4:
        raise ValueError(f"Input tensor must have 4 dimensions (b x l x w x h), but shape is:  {t.shape}")

    b, l, h, w = t.shape

    if l != 3:
        raise ValueError(f"Expected 3 layers, but got: {l}")

    if w % cols != 0:
        raise ValueError(f"Width of image {w} is not divisible by {cols}")

    if h % rows != 0:
        raise ValueError(f"Height of image {h} is not divisible by {rows}")

    cell_width = w // cols
    cell_height = h // rows

    new_tensor = torch.zeros((b, rows * cols, l, cell_height, cell_width))

    for i in range(rows):
        for j in range(cols):
            new_tensor[:, i * cols + j, :, :, :] = t[:, :, i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]

    return new_tensor

def prepare_batch(batch):
    input_batch = torch.stack([elem[0] for elem in batch], 0)
    label_batch = torch.stack([elem[1] for elem in batch], 0)
    return input_batch, label_batch


def show_db_graph():
    db_params = {'host': 'localhost', 'dbname': 'akidb', 'user': 'reinig'}
    database_connection = connect(**db_params)
    tables, column_to_table_map, available_columns = setup_tables(database_connection)
    table_graph = get_table_graph(database_connection, tables)
    nx.draw(table_graph)
    plt.show()

def add_segmentation_mask_to_img(img: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    label_transformed = torch.nn.functional.one_hot(label, num_classes=27).permute(2, 0, 1).bool()
    img_with_segmentation_mask = draw_segmentation_masks(img, label_transformed, colors=get_aki_label_colors(), alpha=0.5)
    return img_with_segmentation_mask

class FilterVoidLabels(transform_lib.Transform):
    def __init__(self, valid_idx: int, void_idx: list[int], ignore_index: int) -> None:
        """Remove void classes from the cityscapes label.
         Supporting legacy and v2 torchvision interface.

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
        with tv_tensors.set_return_type("TVTensor"):
            return image, self.filter(label)


def sample_to_tv_tensor(sample):
    image, label = sample
    return tv_tensors.Image(image), tv_tensors.Mask(label)

def unpack_feature_pyramid(feature_pyramid):
    try:
        # This works for resnets, efficient-nets
        [_, quarter, eights, _, out] = feature_pyramid
    except ValueError:
        # This works for convnexts
        [quarter, eights, _, out] = feature_pyramid
    return quarter, out
