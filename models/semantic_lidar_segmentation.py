import torch
import logging

import torch.nn.functional as F
import torch_geometric.transforms as T
import torchmetrics as tm
import lightning as L

from torch_geometric.nn import MLP, fps, global_max_pool, radius, knn_interpolate
from torch_geometric.nn.conv import PointConv

log = logging.getLogger("rich")

class SetAbstraction(L.LightningModule):
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

class GlobalSetAbstraction(L.LightningModule):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class FeaturePropagation(L.LightningModule):
    def __init__(self, nn, k: int = 3):
        super().__init__()
        self.mlp = MLP(nn)
        self.k = k

    def forward(self, x, pos, new_pos, batch, new_batch):
        # Interpolate features to match new_pos size
        interpolated = knn_interpolate(x, pos, new_pos, batch, new_batch, k=self.k)

        # Update batch size to match new_pos
        return self.mlp(interpolated), new_pos, new_batch


class PointNet2(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        set_abstraction_ratio_1: float = 0.35,
        set_abstraction_ratio_2: float = 0.15,
        set_abstraction_radius_1: float = 0.33,
        set_abstraction_radius_2: float = 0.25,
        dropout: float = 0.1,
        ignore_index: int = 255,
        train_epochs: int = 100,
        data_from_udm: bool = False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.train_epochs = train_epochs
        self.data_from_udm = data_from_udm

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


        log.info(f"PointNet++ - Num classes: {self.num_classes}")
        log.info(f"PointNet++ - Ignore index: {self.ignore_index}")
        log.info(f"PointNet++ - Train epochs: {self.train_epochs}")
        log.info(f"PointNet++ - Data from UnifiedDataModule: {self.data_from_udm}")

    def forward(self, point_clouds):
        batch_size, num_points, _ = point_clouds.size()
        positions = point_clouds.view(-1, 3)
        batch = torch.arange(batch_size, device=positions.device).repeat_interleave(num_points)
        features = positions.clone()

        # Set Abstractions
        sa0_out = (features, positions, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)

        # Feature Propagation
        x, pos, batch = sa2_out
        x, pos, batch = self.fp2_module(x, pos, sa1_out[1], batch, sa1_out[2])
        x, pos, batch = self.fp1_module(x, sa1_out[1], sa0_out[1], batch, sa0_out[2])

        logits = self.mlp(x)

        return logits


    def step(self, batch):
        point_clouds, labels = (batch if not self.data_from_udm else (batch[2], batch[3]))

        logits = self(point_clouds)
        labels = labels.view(-1)

        loss = self.loss_fn(logits, labels)

        return loss, logits, labels 

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        
        train_acc_step = self.train_acc(logits, labels)

        self.log("train_loss", loss)
        self.log("train_accuracy", train_acc_step)

        return dict(loss=loss, logits=logits)

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        val_acc = self.val_acc(logits, labels)

        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_accuracy", val_acc, on_epoch=True, on_step=False, sync_dist=True)
        
        return dict(loss=loss, logits=logits)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        self.test_acc(logits, labels)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_accuracy", self.test_acc, on_step=False, on_epoch=True, sync_dist=True)

        return dict(loss=loss, logits=logits)
    
    def predict_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)

        amt_batch_elems = batch[0].shape[0]
        elems_per_batch = labels.shape[0] // amt_batch_elems  # Also equal to logits.shape[0] // amt_batch_elems

        assert labels.shape[0] % amt_batch_elems == 0
        assert logits.shape[0] % amt_batch_elems == 0
        assert batch[2].shape[0] == batch[3].shape[0]

        logits_list = []
        labels_list = []

        for i in range(amt_batch_elems):
            start_idx = i * elems_per_batch
            end_idx = (i + 1) * elems_per_batch

            elem_logits = logits[start_idx:end_idx]
            elem_labels = labels[start_idx:end_idx]

            # Remove padding
            valid_mask = elem_labels != 255
            elem_logits = elem_logits[valid_mask]
            elem_labels = elem_labels[valid_mask]

            logits_list.append(elem_logits)
            labels_list.append(elem_labels)

        return dict(loss=loss, logits=logits_list, labels=labels_list)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.train_epochs, 1e-7)
        schedule = {
            "scheduler": scheduler,
            "interval": 'epoch',
            "frequency": 1,
        }

        return [[opt], [schedule]] if scheduler else opt
                        