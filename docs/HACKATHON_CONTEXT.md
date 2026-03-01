# Hackathon Context (Persistent Summary)

## Core objective
Train a machine learning model to predict **visceral fat** from **facial images**.

## Constraints and priorities
- This is a hackathon project.
- Time is critical.
- Prioritize rapid prototyping and short iteration loops.
- Build toward a working demo quickly.

## Technical direction
- Primary framework interest: **PyTorch Lightning**.
- Experiment tracking interest: **Weights & Biases (W&B)**.

## Recommended supporting tools
- **Hydra + OmegaConf** for configuration management and reproducible runs.
- **Albumentations** for robust image augmentation.
- **TorchMetrics** for standardized evaluation metrics.
- **Gradio** for fast demo UI deployment.
- **ONNX / ONNX Runtime** (optional) for lightweight inference optimization.
- **pytest** for quick regression checks.
- **pre-commit + ruff + black + isort** for team collaboration and code quality.

## Collaboration setup intent
Use a standard, team-friendly repository layout with clear separation of data, source code, experiments, and demo artifacts.

## Notes for future context windows
When assisting in this repository:
1. Optimize for speed-to-demo.
2. Propose minimal, high-impact baselines first.
3. Keep experiments reproducible and tracked.
4. Prefer simple solutions over over-engineering.
