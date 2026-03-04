from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import lightning as L
import torch
from PIL import Image
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.models.face_to_visceral_lightning import FaceToVisceralRegressor


@dataclass(frozen=True)
class PairRow:
    face_path: Path
    ct_path: Path
    ct_label_path: Path
    ct_patient_id: str
    visceral_ratio: float


class FaceCTPairDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[PairRow],
        image_transform: transforms.Compose,
        target_min: float,
        target_max: float,
    ) -> None:
        self.rows = list(rows)
        self.image_transform = image_transform
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.target_range = (self.target_max - self.target_min) if (self.target_max - self.target_min) > 1e-8 else 1.0

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.image_transform(image)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[index]
        target_raw = torch.tensor(row.visceral_ratio, dtype=torch.float32)
        target_norm_01 = (target_raw - self.target_min) / self.target_range

        face_tensor = self._load_image(row.face_path)
        ct_tensor = self._load_image(row.ct_path)

        return {
            "face": face_tensor,
            "ct": ct_tensor,
            "target_raw": target_raw,
            "target_norm_01": target_norm_01,
            "face_path": str(row.face_path),
            "ct_path": str(row.ct_path),
            "ct_label_path": str(row.ct_label_path),
            "ct_patient_id": row.ct_patient_id,
        }


class FaceCTPairDataModule(L.LightningDataModule):
    def __init__(
        self,
        manifest_path: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.image_size = int(image_size)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)

        self.train_rows: List[PairRow] = []
        self.val_rows: List[PairRow] = []
        self.test_rows: List[PairRow] = []

        self.target_min: float = 0.0
        self.target_max: float = 1.0

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset: FaceCTPairDataset | None = None
        self.val_dataset: FaceCTPairDataset | None = None
        self.test_dataset: FaceCTPairDataset | None = None

    def _read_rows(self) -> List[PairRow]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Pair manifest not found: {self.manifest_path}")

        rows: List[PairRow] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "utk_image_path",
                "ct_image_path",
                "ct_label_path",
                "ct_patient_id",
                "visceral_ratio",
            }
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Pair manifest missing required columns. "
                    f"Found={reader.fieldnames}, required={sorted(required)}"
                )

            for row in reader:
                face_path = Path(row["utk_image_path"])
                ct_path = Path(row["ct_image_path"])
                ct_label_path = Path(row["ct_label_path"])

                if not face_path.exists() or not ct_path.exists() or not ct_label_path.exists():
                    continue

                rows.append(
                    PairRow(
                        face_path=face_path,
                        ct_path=ct_path,
                        ct_label_path=ct_label_path,
                        ct_patient_id=row["ct_patient_id"],
                        visceral_ratio=float(row["visceral_ratio"]),
                    )
                )

        if not rows:
            raise ValueError("No valid rows found in pair manifest")

        return rows

    def _split_by_patient(self, rows: Sequence[PairRow]) -> tuple[List[PairRow], List[PairRow], List[PairRow]]:
        patient_ids = sorted({row.ct_patient_id for row in rows})
        rng = random.Random(self.seed)
        rng.shuffle(patient_ids)

        num_patients = len(patient_ids)
        num_train = int(num_patients * self.train_ratio)
        num_val = int(num_patients * self.val_ratio)

        train_ids = set(patient_ids[:num_train])
        val_ids = set(patient_ids[num_train : num_train + num_val])
        test_ids = set(patient_ids[num_train + num_val :])

        train_rows = [row for row in rows if row.ct_patient_id in train_ids]
        val_rows = [row for row in rows if row.ct_patient_id in val_ids]
        test_rows = [row for row in rows if row.ct_patient_id in test_ids]

        return train_rows, val_rows, test_rows

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None and self.test_dataset is not None:
            return

        rows = self._read_rows()
        self.train_rows, self.val_rows, self.test_rows = self._split_by_patient(rows)

        if not self.train_rows or not self.val_rows or not self.test_rows:
            raise ValueError(
                "Empty split detected. Adjust train_ratio/val_ratio or provide more data rows."
            )

        all_targets = torch.tensor([row.visceral_ratio for row in rows], dtype=torch.float32)
        self.target_min = float(all_targets.min().item())
        self.target_max = float(all_targets.max().item())
        if (self.target_max - self.target_min) < 1e-8:
            self.target_min = 0.0
            self.target_max = 1.0

        self.train_dataset = FaceCTPairDataset(
            rows=self.train_rows,
            image_transform=self.train_transform,
            target_min=self.target_min,
            target_max=self.target_max,
        )
        self.val_dataset = FaceCTPairDataset(
            rows=self.val_rows,
            image_transform=self.eval_transform,
            target_min=self.target_min,
            target_max=self.target_max,
        )
        self.test_dataset = FaceCTPairDataset(
            rows=self.test_rows,
            image_transform=self.eval_transform,
            target_min=self.target_min,
            target_max=self.target_max,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting val_dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting test_dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Lightning model to predict visceral fat ratio from paired face+CT images, "
            "using global min-max normalized targets in [0,1] and MSE loss."
        )
    )
    parser.add_argument(
        "--pair-manifest",
        type=Path,
        default=Path("data/paired_face_ct/paired_manifest.csv"),
        help="Path to paired face-CT manifest generated by create_face_ct_pairs_by_age_bins.py",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/face_to_visceral"),
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable pretrained ImageNet backbones",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="face2visceral",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team (optional)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (optional)",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    return parser.parse_args()


def resolve_trainer_accelerator() -> tuple[str, int]:
    if torch.backends.mps.is_available():
        return "mps", 1
    if torch.cuda.is_available():
        return "cuda", 1
    return "cpu", 1


def main() -> None:
    args = parse_args()

    L.seed_everything(args.seed, workers=True)

    data_module = FaceCTPairDataModule(
        manifest_path=args.pair_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    data_module.setup("fit")

    model = FaceToVisceralRegressor(
        lr=args.lr,
        weight_decay=args.weight_decay,
        target_min=data_module.target_min,
        target_max=data_module.target_max,
        pretrained_backbone=not args.no_pretrained,
        image_size=args.image_size,
    )

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_rmse_raw",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=checkpoint_dir,
        filename="face2visceral-{epoch:02d}-{val_rmse_raw:.5f}",
    )

    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="val_rmse_raw",
        mode="min",
        patience=8,
    )

    loggers = [CSVLogger(save_dir=str(args.output_dir), name="csv_logs")]
    wandb_logger: WandbLogger | None = None
    if not args.disable_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            save_dir=str(args.output_dir),
            log_model=False,
        )
        wandb_logger.log_hyperparams(vars(args))
        loggers.append(wandb_logger)

    accelerator, devices = resolve_trainer_accelerator()

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=str(args.output_dir),
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        logger=loggers,
        accelerator=accelerator,
        devices=devices,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")

    if wandb_logger is not None:
        wandb_logger.experiment.finish()

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Target normalization stats (whole dataset): min={data_module.target_min:.8f}, max={data_module.target_max:.8f}")


if __name__ == "__main__":
    main()
