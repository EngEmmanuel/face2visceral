from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.models.face_to_visceral_lightning import FaceToVisceralRegressor


def collect_image_paths(input_path: Path) -> List[Path]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if input_path.is_file():
        return [input_path]

    paths = [p for p in sorted(input_path.rglob("*")) if p.is_file() and p.suffix.lower() in valid_ext]
    if not paths:
        raise ValueError(f"No image files found under: {input_path}")
    return paths


def collect_pairs(face_input: Path, ct_input: Path) -> List[Tuple[Path, Path]]:
    face_paths = collect_image_paths(face_input)
    ct_paths = collect_image_paths(ct_input)

    if len(face_paths) != len(ct_paths):
        raise ValueError(
            f"Mismatched image counts: faces={len(face_paths)}, ct={len(ct_paths)}. "
            "Provide equal counts and aligned ordering."
        )

    return list(zip(face_paths, ct_paths))


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict visceral fat ratio from paired face + CT image inputs using a trained Lightning checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--face-input",
        type=Path,
        required=True,
        help="Path to a face image file or a directory of face images",
    )
    parser.add_argument(
        "--ct-input",
        type=Path,
        required=True,
        help="Path to a CT image file or a directory of CT images",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/face_to_visceral/predictions.csv"),
        help="Where to save prediction CSV",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()

    model = FaceToVisceralRegressor.load_from_checkpoint(
        checkpoint_path=str(args.checkpoint),
        map_location="cpu",
    )
    model.eval()

    device = resolve_device(args.device)
    model.to(device)

    image_size = int(model.hparams.image_size)
    transform = build_transform(image_size=image_size)

    pairs = collect_pairs(args.face_input, args.ct_input)

    predictions: List[dict] = []

    with torch.inference_mode():
        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start : start + args.batch_size]
            face_tensors = []
            ct_tensors = []

            for face_path, ct_path in batch_pairs:
                face_img = Image.open(face_path).convert("RGB")
                ct_img = Image.open(ct_path).convert("RGB")
                face_tensors.append(transform(face_img))
                ct_tensors.append(transform(ct_img))

            face_batch = torch.stack(face_tensors, dim=0).to(device)
            ct_batch = torch.stack(ct_tensors, dim=0).to(device)
            pred_norm_01 = model(face_batch, ct_batch).view(-1)
            pred_ratio = model.denormalize_target_01(pred_norm_01).view(-1).detach().cpu().tolist()
            pred_norm_01_list = pred_norm_01.detach().cpu().tolist()

            for (face_path, ct_path), norm_value, raw_value in zip(batch_pairs, pred_norm_01_list, pred_ratio):
                predictions.append(
                    {
                        "face_path": str(face_path.resolve()),
                        "ct_path": str(ct_path.resolve()),
                        "pred_visceral_ratio_norm_01": f"{float(norm_value):.8f}",
                        "pred_visceral_ratio": f"{float(raw_value):.8f}",
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["face_path", "ct_path", "pred_visceral_ratio_norm_01", "pred_visceral_ratio"],
        )
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
