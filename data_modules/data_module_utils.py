import torch
import torchvision as tv
import torchvision.transforms.v2 as transform_lib

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
  