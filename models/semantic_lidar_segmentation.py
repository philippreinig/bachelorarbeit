
import torch
import logging

import torch.nn.functional as F
import torch_geometric.transforms as T
import torchmetrics as tm

from lightning.pytorch import LightningModule

from torch_geometric.nn import MLP, fps, global_max_pool, radius, knn_interpolate
from torch_geometric.nn.conv import PointConv

log = logging.getLogger("rich")

class SetAbstraction(LightningModule):
    def __init__(self, ratio, radius, nn):
        super().__init__()
        self.ratio = ratio
        self.radius = radius
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSetAbstraction(LightningModule):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FeaturePropagation(LightningModule):
    def __init__(self, nn, k: int = 3):
        super().__init__()
        self.mlp = MLP(nn)
        self.k = k

    def forward(self, x, pos, new_pos, batch, new_batch):
        # Ensure batch sizes match point sizes
        assert pos.size(0) == batch.size(0), f"pos size {pos.size(0)} != batch size {batch.size(0)}"
        assert new_pos.size(0) == new_batch.size(0), f"new_pos size {new_pos.size(0)} != new_batch size {new_batch.size(0)}"

        # Interpolate features to match new_pos size
        interpolated = knn_interpolate(x, pos, new_pos, batch, new_batch, k=self.k)

        # Ensure batch size matches interpolated size
        assert interpolated.size(0) == new_batch.size(0), f"Interpolated size {interpolated.size(0)} != new_batch size {new_batch.size(0)}"

        # Update batch size to match new_pos
        return self.mlp(interpolated), new_pos, new_batch


class PointNet2(LightningModule):
    def __init__(
        self,
        num_classes: int,
        set_abstraction_ratio_1: float = 0.75,
        set_abstraction_ratio_2: float = 0.5,
        set_abstraction_radius_1: float = 0.33,
        set_abstraction_radius_2: float = 0.25,
        dropout: float = 0.1,
        ignore_index: int = 255,
        train_epochs: int = 30,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.train_epochs = train_epochs

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            MLP([3 + 3, 64, 64, 128])
        )
        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            MLP([128 + 3, 128, 128, 256])
        ) 
        #self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))

        self.fp2_module = FeaturePropagation([256, 256, 128])
        self.fp1_module = FeaturePropagation([128, 128, 64])

        self.mlp = MLP([64, 32, self.num_classes], dropout=dropout, norm=None)

        self.train_acc = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average="micro")
        self.val_acc = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average="micro")
        self.test_acc = tm.classification.Accuracy(task="multiclass", num_classes=self.num_classes, ignore_index=self.ignore_index, average="micro")

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, point_clouds):
        log.info(f"Memory allocated at start of forward: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

        batch_size, num_points, _ = point_clouds.size()
        positions = point_clouds.view(-1, 3)
        batch = torch.arange(batch_size, device=positions.device).repeat_interleave(num_points)
        features = positions.clone()

        # Set Abstractions
        sa0_out = (features, positions, batch)
        log.info(f"Before sa1_module: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        sa1_out = self.sa1_module(*sa0_out)

        log.info(f"Before sa2_module: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        sa2_out = self.sa2_module(*sa1_out)

        # Feature Propagation
        x, pos, batch = sa2_out
        log.info(f"Before fp2_module: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        x, pos, batch = self.fp2_module(x, pos, sa1_out[1], batch, sa1_out[2])

        log.info(f"Before fp1_module: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        x, pos, batch = self.fp1_module(x, sa1_out[1], sa0_out[1], batch, sa0_out[2])

        log.info(f"Before MLP: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        logits = self.mlp(x)

        log.info(f"End of forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

        return logits


    def step(self, batch):
        point_clouds, labels = batch
        log.info(f"Memory allocated before forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

        logits = self(point_clouds)
        labels = labels.view(-1)

        log.info(f"Memory allocated after forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

        loss = self.loss_fn(logits, labels)
        log.info(f"Memory allocated after loss calculation: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

        return loss, logits, labels 

    def training_step(self, batch, batch_idx):
        log.info(f"Training step {batch_idx}")
        loss, logits, labels = self.step(batch)
        
        self.train_acc(logits, labels)

        self.log("train/loss", loss)
        self.log("train/accuracy", self.train_acc)

        log.info(f"Loss and accuracy after training step {batch_idx}: {loss}, {self.train_acc}")

        return dict(loss=loss, logits=logits)

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        self.val_acc(logits, labels)

        self.log("validation/loss", loss)
        self.log("validation/accuracy", self.val_acc)
        
        return dict(loss=loss, logits=logits)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        self.test_acc(logits, labels)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/accuracy", self.test_acc)

        return dict(loss=loss, logits=logits)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.train_epochs, 1e-7)
        schedule = {
            "scheduler": scheduler,
            "interval": 'epoch',
            "frequency": 1,
        }

        return [[opt], [schedule]] if scheduler else opt
                        