from __future__ import annotations

from typing import Dict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _build_resnet18_backbone(pretrained: bool = True) -> nn.Module:
    if pretrained:
        try:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (RuntimeError, OSError, ValueError):
            model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)

    model.fc = nn.Identity()
    return model


class FaceToVisceralRegressor(L.LightningModule):
    """Lightning model for visceral fat ratio regression from paired face + CT inputs.

    Target is the global min-max normalized visceral ratio in [0, 1], optimized with MSE.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        target_min: float = 0.0,
        target_max: float = 1.0,
        pretrained_backbone: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        _ = (
            lr,
            weight_decay,
            target_min,
            target_max,
            image_size,
        )
        self.save_hyperparameters()

        self.face_encoder = _build_resnet18_backbone(pretrained=pretrained_backbone)
        self.ct_encoder = _build_resnet18_backbone(pretrained=pretrained_backbone)

        self.fused_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1),
        )

    @property
    def target_min(self) -> float:
        return float(self.hparams.target_min)

    @property
    def target_max(self) -> float:
        return float(self.hparams.target_max)

    @property
    def target_range(self) -> float:
        value = self.target_max - self.target_min
        return value if value > 1e-8 else 1.0

    def denormalize_target_01(self, y_norm_01: torch.Tensor) -> torch.Tensor:
        return y_norm_01 * self.target_range + self.target_min

    @staticmethod
    def unmap_unit_interval(
        y_norm_01: torch.Tensor,
        out_min: float,
        out_max: float,
    ) -> torch.Tensor:
        out_range = out_max - out_min
        return y_norm_01 * out_range + out_min

    def unmap_to_10_50(self, y_norm_01: torch.Tensor) -> torch.Tensor:
        return self.unmap_unit_interval(y_norm_01=y_norm_01, out_min=10.0, out_max=50.0)

    def normalize_target_to_01(self, y_raw: torch.Tensor) -> torch.Tensor:
        return (y_raw - self.target_min) / self.target_range

    def forward(self, face: torch.Tensor, ct: torch.Tensor) -> torch.Tensor:
        face_features = self.face_encoder(face)
        ct_features = self.ct_encoder(ct)
        fused_features = torch.cat([face_features, ct_features], dim=1)
        pred_norm_01 = torch.sigmoid(self.fused_head(fused_features))
        return pred_norm_01

    def _common_metrics(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        face = batch["face"]
        ct = batch["ct"]
        y_norm_01 = batch["target_norm_01"].view(-1, 1)
        y_raw = batch["target_raw"].view(-1, 1)

        pred_norm_01 = self.forward(face, ct)
        loss = F.mse_loss(pred_norm_01, y_norm_01)

        pred_raw = self.denormalize_target_01(pred_norm_01)
        mse_raw = F.mse_loss(pred_raw, y_raw)
        rmse_raw = torch.sqrt(mse_raw)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_rmse_raw", rmse_raw, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        _ = batch_idx
        face = batch["face"]
        ct = batch["ct"]
        y_norm_01 = batch["target_norm_01"].view(-1, 1)

        pred_norm_01 = self.forward(face, ct)
        total_loss = F.mse_loss(pred_norm_01, y_norm_01)

        pred_raw = self.denormalize_target_01(pred_norm_01)
        y_raw = batch["target_raw"].view(-1, 1)
        rmse_raw = torch.sqrt(F.mse_loss(pred_raw, y_raw))

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_rmse_raw", rmse_raw, prog_bar=True, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
        return self._common_metrics(batch=batch, stage="val")

    def test_step(self, batch, batch_idx):
        _ = batch_idx
        return self._common_metrics(batch=batch, stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _ = (batch_idx, dataloader_idx)
        face = batch["face"]
        ct = batch["ct"]
        pred_norm_01 = self.forward(face, ct).view(-1)
        pred_raw = self.denormalize_target_01(pred_norm_01).view(-1)

        return {
            "pred_visceral_ratio_norm_01": pred_norm_01.detach().cpu(),
            "pred_visceral_ratio": pred_raw.detach().cpu(),
            "face_path": batch.get("face_path", None),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,
            eta_min=float(self.hparams.lr) * 0.05,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


__all__ = ["FaceToVisceralRegressor"]
