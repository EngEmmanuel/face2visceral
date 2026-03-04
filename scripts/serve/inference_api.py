from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def ratio_to_white_pixels(ratio: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    image_area = float(image_height * image_width)
    return ratio * image_area


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_checkpoint(checkpoint_path: Optional[Path]) -> Path:
    if checkpoint_path is not None:
        resolved = checkpoint_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved

    ckpt_dir = (REPO_ROOT / "artifacts" / "face_to_visceral" / "checkpoints").resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # Prefer best-named checkpoints generated during training.
    candidates = sorted(
        [p for p in ckpt_dir.glob("face2visceral-*.ckpt")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"No training checkpoints found in: {ckpt_dir}. "
        "Provide an explicit path with --checkpoint."
    )


def build_transform(image_size: int) -> Callable[[Image.Image], torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _transform(image: Image.Image) -> torch.Tensor:
        resized = image.resize((image_size, image_size), resample=2)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr[:, :, :3]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - mean) / std

    return _transform


def create_app(checkpoint_path: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="Face2Visceral Inference API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.device = resolve_device()
    app.state.model = None
    app.state.transform = None
    app.state.checkpoint = find_checkpoint(checkpoint_path)
    app.state.model_mode = "checkpoint"

    @app.on_event("startup")
    def load_model_once() -> None:
        from scripts.models.face_to_visceral_lightning import FaceToVisceralRegressor

        model_cls: Any = FaceToVisceralRegressor
        loader = getattr(model_cls, "load_from_checkpoint")
        model = loader(
            checkpoint_path=str(app.state.checkpoint),
            map_location="cpu",
            strict=False,
        )
        model.eval()
        model.to(app.state.device)

        image_size = int(model.hparams.image_size)
        app.state.transform = build_transform(image_size)
        app.state.model = model

        print(f"[startup] Loaded checkpoint: {app.state.checkpoint}")
        print(f"[startup] Using device: {app.state.device}")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": str(app.state.device),
            "checkpoint": str(app.state.checkpoint) if app.state.checkpoint else None,
            "mode": app.state.model_mode,
        }

    @app.post("/predict")
    async def predict(
        image: UploadFile = File(...),
        age: Optional[str] = Form(default=None),
        sex: Optional[str] = Form(default=None),
    ) -> dict:
        if app.state.model is None or app.state.transform is None:
            raise HTTPException(status_code=503, detail="Model not ready")

        try:
            payload = await image.read()
            pil_image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

        original_width, original_height = pil_image.size
        image_area = int(original_height * original_width)

        tensor = app.state.transform(pil_image).unsqueeze(0).to(app.state.device)

        with torch.inference_mode():
            pred_norm_01 = app.state.model(tensor, tensor).view(-1)
            pred_ratio = app.state.model.denormalize_target_01(pred_norm_01).view(-1)

            pred_white_pixels = ratio_to_white_pixels(
                ratio=pred_ratio,
                image_height=original_height,
                image_width=original_width,
            ).view(-1)

        ratio_value = float(pred_ratio.item())
        ratio_clamped = max(0.0, min(1.0, ratio_value))
        percent = ratio_clamped * 100.0

        return {
            "prediction": f"{percent:.2f}%",
            "result": f"{percent:.2f}%",
            "fat_ratio": ratio_value,
            "fat_ratio_normalized_01": float(pred_norm_01.item()),
            "estimated_white_pixels": float(pred_white_pixels.item()),
            "image_area": image_area,
            "mode": app.state.model_mode,
            "input_age": age,
            "input_sex": sex,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Face2Visceral inference API")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional explicit checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import uvicorn

    uvicorn.run(
        create_app(checkpoint_path=args.checkpoint),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
